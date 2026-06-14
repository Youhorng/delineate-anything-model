#!/usr/bin/env python3
"""Merge tile KMZ/KML parts, clean polygons, and renumber field IDs from 1."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}

ID_FIELD_PATTERNS = [
    re.compile(r"^id_\d+$", re.I),
    re.compile(r"^id\d*$", re.I),
    re.compile(r"^field_id$", re.I),
    re.compile(r"^fid$", re.I),
]

NEW_NAME_PATTERN = re.compile(r"^new\d+$", re.I)
NEW_SPACE_PATTERN = re.compile(r"^new\s+\d+$", re.I)
NUMERIC_NAME_PATTERN = re.compile(r"^\d+$")


def read_kml_bytes(path: Path) -> bytes:
    if path.suffix.lower() == ".kmz":
        with zipfile.ZipFile(path) as zf:
            kml_names = [name for name in zf.namelist() if name.lower().endswith(".kml")]
            if not kml_names:
                raise ValueError(f"No KML file found inside KMZ: {path}")
            preferred = "doc.kml" if "doc.kml" in kml_names else kml_names[0]
            return zf.read(preferred)
    return path.read_bytes()


def parse_coordinate_tokens(text: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for token in text.strip().split():
        parts = token.split(",")
        if len(parts) < 2:
            continue
        points.append((float(parts[0]), float(parts[1])))
    return points


def polygon_from_linear_rings(outer: list[tuple[float, float]], inners: list[list[tuple[float, float]]]) -> Polygon | None:
    if len(outer) < 4:
        return None
    if outer[0] != outer[-1]:
        outer = outer + [outer[0]]
    holes = []
    for inner in inners:
        if len(inner) < 4:
            continue
        if inner[0] != inner[-1]:
            inner = inner + [inner[0]]
        holes.append(inner)
    return Polygon(outer, holes)


def geometry_from_placemark(placemark: ET.Element) -> Polygon | MultiPolygon | None:
    polygon_el = placemark.find("kml:Polygon", KML_NS)
    if polygon_el is not None:
        outer_el = polygon_el.find("kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", KML_NS)
        if outer_el is None or not outer_el.text:
            return None
        outer = parse_coordinate_tokens(outer_el.text)
        inners = []
        for inner_el in polygon_el.findall("kml:innerBoundaryIs/kml:LinearRing/kml:coordinates", KML_NS):
            if inner_el.text:
                inners.append(parse_coordinate_tokens(inner_el.text))
        return polygon_from_linear_rings(outer, inners)

    multi_el = placemark.find("kml:MultiGeometry", KML_NS)
    if multi_el is not None:
        polygons: list[Polygon] = []
        for child in multi_el.findall("kml:Polygon", KML_NS):
            outer_el = child.find("kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", KML_NS)
            if outer_el is None or not outer_el.text:
                continue
            outer = parse_coordinate_tokens(outer_el.text)
            inners = []
            for inner_el in child.findall("kml:innerBoundaryIs/kml:LinearRing/kml:coordinates", KML_NS):
                if inner_el.text:
                    inners.append(parse_coordinate_tokens(inner_el.text))
            poly = polygon_from_linear_rings(outer, inners)
            if poly is not None:
                polygons.append(poly)
        if not polygons:
            return None
        if len(polygons) == 1:
            return polygons[0]
        return MultiPolygon(polygons)

    coords_el = placemark.find(".//kml:coordinates", KML_NS)
    if coords_el is not None and coords_el.text:
        outer = parse_coordinate_tokens(coords_el.text)
        return polygon_from_linear_rings(outer, [])
    return None


def extract_attrs(placemark: ET.Element) -> dict[str, Any]:
    attrs: dict[str, Any] = {}

    schema = placemark.find("kml:ExtendedData/kml:SchemaData", KML_NS)
    if schema is not None:
        for simple_data in schema.findall("kml:SimpleData", KML_NS):
            field_name = simple_data.attrib.get("name")
            if field_name:
                attrs[field_name] = simple_data.text

    for data_el in placemark.findall("kml:ExtendedData/kml:Data", KML_NS):
        field_name = data_el.attrib.get("name")
        value_el = data_el.find("kml:value", KML_NS)
        if field_name and value_el is not None and value_el.text is not None:
            attrs[field_name] = value_el.text

    return attrs


def coerce_numeric(value: Any) -> int | float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number.is_integer():
        return int(number)
    return number


def detect_id_field(attr_names: set[str], requested: str) -> str | None:
    if requested and requested.lower() != "auto":
        if requested not in attr_names:
            raise ValueError(
                f"Requested id field '{requested}' not found. Available: {sorted(attr_names)}"
            )
        return requested

    ranked: list[tuple[int, str]] = []
    for name in attr_names:
        for index, pattern in enumerate(ID_FIELD_PATTERNS):
            if pattern.match(name):
                ranked.append((index, name))
                break
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked[0][1]


def classify_feature(name: str | None, attrs: dict[str, Any], id_field: str | None) -> tuple[str, int | None]:
    if id_field and id_field in attrs:
        numeric_id = coerce_numeric(attrs[id_field])
        if isinstance(numeric_id, int):
            return "original", numeric_id

    if name and NUMERIC_NAME_PATTERN.match(name.strip()):
        return "original", int(name.strip())

    if name and (
        NEW_NAME_PATTERN.match(name.strip())
        or NEW_SPACE_PATTERN.match(name.strip())
    ):
        return "added", None

    return "added", None


def clean_geometry(geom: Polygon | MultiPolygon) -> tuple[Polygon | MultiPolygon | None, bool]:
    changed = False
    if geom.is_empty:
        return None, changed

    if not geom.is_valid:
        geom = make_valid(geom)
        changed = True

    if geom.geom_type == "GeometryCollection":
        polygons = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not polygons:
            return None, changed
        geom = polygons[0]
        changed = True

    if geom.geom_type == "Polygon":
        geom = Polygon([(x, y) for x, y, *_ in geom.exterior.coords], holes=[
            [(x, y) for x, y, *_ in ring.coords]
            for ring in geom.interiors
        ])
    elif geom.geom_type == "MultiPolygon":
        geom = MultiPolygon([
            Polygon([(x, y) for x, y, *_ in poly.exterior.coords], holes=[
                [(x, y) for x, y, *_ in ring.coords]
                for ring in poly.interiors
            ])
            for poly in geom.geoms
        ])

    if geom.is_empty or geom.area == 0:
        return None, changed
    return geom, changed


def count_polygon_vertices(geom: Polygon | MultiPolygon) -> int:
    if geom.geom_type == "Polygon":
        total = len(geom.exterior.coords) - 1
        for interior in geom.interiors:
            total += len(interior.coords) - 1
        return total
    return sum(count_polygon_vertices(poly) for poly in geom.geoms)


def simplify_geometry(
    geom: Polygon | MultiPolygon,
    tolerance_m: float,
    utm_epsg: int = 32648,
) -> Polygon | MultiPolygon | None:
    if tolerance_m <= 0:
        return geom

    series = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(epsg=utm_epsg)
    simplified = series.simplify(tolerance_m, preserve_topology=True).iloc[0]
    if simplified is None or simplified.is_empty:
        return None

    result = (
        gpd.GeoSeries([simplified], crs=utm_epsg)
        .to_crs("EPSG:4326")
        .iloc[0]
    )
    if result is None or result.is_empty:
        return None

    cleaned, _ = clean_geometry(result)
    return cleaned


def parse_vector_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".geojson":
        gdf = gpd.read_file(path)
        features: list[dict[str, Any]] = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            attrs = {
                col: row[col]
                for col in gdf.columns
                if col != "geometry" and pd.notna(row[col])
            }
            features.append(
                {
                    "name": attrs.pop("name", None),
                    "style_url": attrs.pop("style_url", None),
                    "attrs": attrs,
                    "geometry": geom,
                    "source_file": path.name,
                }
            )
        return features

    root = ET.fromstring(read_kml_bytes(path))
    features = []
    for placemark in root.findall(".//kml:Placemark", KML_NS):
        name_el = placemark.find("kml:name", KML_NS)
        style_el = placemark.find("kml:styleUrl", KML_NS)
        geom = geometry_from_placemark(placemark)
        if geom is None:
            continue

        features.append(
            {
                "name": name_el.text.strip() if name_el is not None and name_el.text else None,
                "style_url": style_el.text if style_el is not None else None,
                "attrs": extract_attrs(placemark),
                "geometry": geom,
                "source_file": path.name,
            }
        )
    return features


def load_inputs(paths: list[Path]) -> tuple[list[dict[str, Any]], set[str]]:
    all_features: list[dict[str, Any]] = []
    attr_names: set[str] = set()
    for path in paths:
        features = parse_vector_file(path)
        all_features.extend(features)
        for feature in features:
            attr_names.update(feature["attrs"].keys())
    return all_features, attr_names


def choose_candidate(
    current: dict[str, Any],
    challenger: dict[str, Any],
    prefer: str,
) -> dict[str, Any]:
    if prefer == "first":
        return current
    if prefer == "last":
        return challenger

    area_current = current["geometry"].area
    area_challenger = challenger["geometry"].area
    return challenger if area_challenger > area_current else current


def spatial_key(geom: Polygon | MultiPolygon, precision: int = 5) -> tuple[float, float]:
    centroid = geom.centroid
    return (round(centroid.x, precision), round(centroid.y, precision))


def merge_features(
    features: list[dict[str, Any]],
    id_field: str | None,
    prefer: str,
    dedupe_spatial: bool,
    drop_added_overlapping_original: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    originals: dict[int, dict[str, Any]] = {}
    added: list[dict[str, Any]] = []
    invalid_fixed = 0
    dropped_invalid = 0
    original_candidate_count = 0

    for feature in features:
        geom, changed = clean_geometry(feature["geometry"])
        if changed:
            invalid_fixed += 1
        if geom is None:
            dropped_invalid += 1
            continue

        feature = {**feature, "geometry": geom}
        merge_class, legacy_id = classify_feature(feature["name"], feature["attrs"], id_field)
        feature["merge_class"] = merge_class
        feature["legacy_id"] = legacy_id
        feature["legacy_name"] = feature["name"]

        if merge_class == "original" and legacy_id is not None:
            original_candidate_count += 1
            existing = originals.get(legacy_id)
            if existing is None:
                originals[legacy_id] = feature
            else:
                originals[legacy_id] = choose_candidate(existing, feature, prefer)
        else:
            added.append(feature)

    if dedupe_spatial:
        seen = set()
        deduped_added = []
        removed = 0
        for feature in added:
            key = spatial_key(feature["geometry"])
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            deduped_added.append(feature)
        added = deduped_added
    else:
        removed = 0

    original_geoms = list(originals.values())
    if drop_added_overlapping_original and original_geoms:
        original_keys = {spatial_key(f["geometry"]) for f in original_geoms}
        filtered_added = []
        removed_overlap = 0
        for feature in added:
            if spatial_key(feature["geometry"]) in original_keys:
                removed_overlap += 1
                continue
            filtered_added.append(feature)
        added = filtered_added
    else:
        removed_overlap = 0

    merged = original_geoms + added
    stats = {
        "original_ids_union": len(originals),
        "duplicate_original_ids_resolved": max(0, original_candidate_count - len(originals)),
        "added_polygons_collected": len(added) + removed + removed_overlap,
        "added_spatial_duplicates_removed": removed,
        "added_overlapping_original_removed": removed_overlap,
        "invalid_geometries_fixed": invalid_fixed,
        "invalid_geometries_dropped": dropped_invalid,
        "merged_before_renumber": len(merged),
    }
    return merged, stats


def renumber_features(
    features: list[dict[str, Any]],
    id_field: str | None,
    renumber_from: int,
    keep_legacy_ids: bool,
) -> list[dict[str, Any]]:
    def sort_key(feature: dict[str, Any]) -> tuple[float, float]:
        centroid = feature["geometry"].centroid
        return (-centroid.y, centroid.x)

    sorted_features = sorted(features, key=sort_key)
    output = []
    next_id = renumber_from
    for feature in sorted_features:
        row = {
            "field_id": next_id,
            "name": str(next_id),
            "merge_class": feature["merge_class"],
            "merge_source": feature["source_file"],
            "geometry": feature["geometry"],
        }
        if keep_legacy_ids:
            row["legacy_id"] = feature.get("legacy_id")
            row["legacy_name"] = feature.get("legacy_name")

        for key, value in feature.get("attrs", {}).items():
            if key == id_field:
                continue
            row[key] = value

        if id_field:
            row[id_field] = next_id

        output.append(row)
        next_id += 1
    return output


def features_to_gdf(features: list[dict[str, Any]]) -> gpd.GeoDataFrame:
    rows = []
    geometries = []
    for feature in features:
        geom = feature.pop("geometry")
        rows.append(feature)
        geometries.append(geom)
    return gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")


def infer_output_path(inputs: list[Path], output: Path | None) -> Path:
    if output is not None:
        return output
    stem = inputs[0].stem
    for suffix in ("_top", "_bottom", "_left", "_right"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return inputs[0].parent / f"{stem}_merged.geojson"


def build_report(
    inputs: list[Path],
    id_field: str | None,
    merge_stats: dict[str, Any],
    final_count: int,
    renumber_from: int,
) -> dict[str, Any]:
    return {
        "input_files": [str(path) for path in inputs],
        "id_field_detected": id_field,
        "renumbered_from": renumber_from,
        "renumbered_to": renumber_from + final_count - 1 if final_count else None,
        "final_count": final_count,
        **merge_stats,
    }


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    inputs = list(args.inputs or [])
    if args.top:
        inputs.append(args.top)
    if args.bottom:
        inputs.append(args.bottom)
    if args.left:
        inputs.append(args.left)
    if args.right:
        inputs.append(args.right)

    deduped: list[Path] = []
    seen = set()
    for path in inputs:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    if not deduped:
        raise SystemExit("Provide at least one input via --inputs and/or --top/--bottom/--left/--right.")
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge tile KMZ/KML parts, clean polygons, and renumber IDs from 1."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[],
        help="One or more KMZ/KML/GeoJSON part files",
    )
    parser.add_argument("--top", type=Path, help="Optional top part file")
    parser.add_argument("--bottom", type=Path, help="Optional bottom part file")
    parser.add_argument("--left", type=Path, help="Optional left part file")
    parser.add_argument("--right", type=Path, help="Optional right part file")
    parser.add_argument("--output", type=Path, default=None, help="Output GeoJSON path")
    parser.add_argument(
        "--id-field",
        default="auto",
        help="ID attribute name (e.g. id_2, id_3) or 'auto' (default)",
    )
    parser.add_argument(
        "--prefer",
        choices=["first", "last", "largest"],
        default="last",
        help="When the same numeric ID appears in multiple files (default: last)",
    )
    parser.add_argument(
        "--renumber-from",
        type=int,
        default=1,
        help="Start numbering merged fields from this integer (default: 1)",
    )
    parser.add_argument(
        "--no-renumber",
        action="store_true",
        help="Skip final renumbering step",
    )
    parser.add_argument(
        "--no-keep-legacy-ids",
        action="store_true",
        help="Do not write legacy_id / legacy_name columns",
    )
    parser.add_argument(
        "--dedupe-spatial",
        action="store_true",
        help="Drop added polygons that share the same centroid location",
    )
    parser.add_argument(
        "--drop-added-overlapping-original",
        action="store_true",
        help="Drop added polygons whose centroid matches an original field centroid",
    )
    parser.add_argument("--report", type=Path, default=None, help="Merge report JSON path")
    args = parser.parse_args()

    inputs = resolve_inputs(args)
    features, attr_names = load_inputs(inputs)
    id_field = detect_id_field(attr_names, args.id_field)

    merged, merge_stats = merge_features(
        features,
        id_field=id_field,
        prefer=args.prefer,
        dedupe_spatial=args.dedupe_spatial,
        drop_added_overlapping_original=args.drop_added_overlapping_original,
    )

    if not args.no_renumber:
        merged_rows = renumber_features(
            merged,
            id_field=id_field,
            renumber_from=args.renumber_from,
            keep_legacy_ids=not args.no_keep_legacy_ids,
        )
    else:
        merged_rows = []
        for feature in merged:
            row = {
                "name": feature.get("name"),
                "merge_class": feature["merge_class"],
                "merge_source": feature["source_file"],
                "geometry": feature["geometry"],
            }
            if not args.no_keep_legacy_ids:
                row["legacy_id"] = feature.get("legacy_id")
                row["legacy_name"] = feature.get("legacy_name")
            row.update(feature.get("attrs", {}))
            merged_rows.append(row)

    gdf = features_to_gdf(merged_rows)
    output_path = infer_output_path(inputs, args.output)
    report_path = args.report or output_path.with_suffix(".merge_report.json")

    gdf.to_file(output_path, driver="GeoJSON")
    report = build_report(inputs, id_field, merge_stats, len(gdf), args.renumber_from)
    report["output_geojson"] = str(output_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote merged GeoJSON: {output_path}")
    print(f"Wrote merge report:   {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
