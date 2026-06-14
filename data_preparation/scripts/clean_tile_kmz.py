#!/usr/bin/env python3
"""Clean a single tile KMZ/KML: fix geometries and renumber field IDs from 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from merge_tile_kmz import (
    choose_candidate,
    classify_feature,
    clean_geometry,
    count_polygon_vertices,
    detect_id_field,
    features_to_gdf,
    parse_vector_file,
    renumber_features,
    simplify_geometry,
    spatial_key,
)


def cleanup_features(
    features: list[dict[str, Any]],
    id_field: str | None,
    prefer: str,
    dedupe_spatial: bool,
    drop_added_overlapping_original: bool,
    simplify_tolerance_m: float | None = None,
    utm_epsg: int = 32648,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    originals: dict[int, dict[str, Any]] = {}
    added: list[dict[str, Any]] = []
    invalid_fixed = 0
    dropped_invalid = 0
    original_candidate_count = 0
    vertices_before = 0
    vertices_after = 0
    simplified_count = 0
    dropped_after_simplify = 0

    for feature in features:
        geom, changed = clean_geometry(feature["geometry"])
        if changed:
            invalid_fixed += 1
        if geom is None:
            dropped_invalid += 1
            continue

        if simplify_tolerance_m is not None and simplify_tolerance_m > 0:
            before = count_polygon_vertices(geom)
            vertices_before += before
            geom = simplify_geometry(geom, simplify_tolerance_m, utm_epsg=utm_epsg)
            if geom is None:
                dropped_after_simplify += 1
                continue
            after = count_polygon_vertices(geom)
            vertices_after += after
            if after < before:
                simplified_count += 1
        elif simplify_tolerance_m is not None:
            vertices_before += count_polygon_vertices(geom)
            vertices_after += count_polygon_vertices(geom)

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

    added_before_dedupe = len(added)
    if dedupe_spatial:
        seen: set[tuple[float, float]] = set()
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
        original_keys = {spatial_key(feature["geometry"]) for feature in original_geoms}
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
        "input_polygon_count": len(features),
        "original_ids_unique": len(originals),
        "duplicate_original_ids_resolved": max(0, original_candidate_count - len(originals)),
        "added_polygon_count": added_before_dedupe,
        "added_spatial_duplicates_removed": removed,
        "added_overlapping_original_removed": removed_overlap,
        "invalid_geometries_fixed": invalid_fixed,
        "invalid_geometries_dropped": dropped_invalid,
        "before_renumber": len(merged),
        "simplify_tolerance_m": simplify_tolerance_m,
        "utm_epsg": utm_epsg if simplify_tolerance_m else None,
        "polygons_simplified": simplified_count if simplify_tolerance_m else 0,
        "vertices_before_simplify": vertices_before if simplify_tolerance_m else None,
        "vertices_after_simplify": vertices_after if simplify_tolerance_m else None,
        "dropped_after_simplify": dropped_after_simplify if simplify_tolerance_m else 0,
    }
    return merged, stats


def infer_output_path(input_path: Path, output: Path | None) -> Path:
    if output is not None:
        return output
    return input_path.with_name(f"{input_path.stem}_cleaned.geojson")


def build_report(
    input_path: Path,
    id_field: str | None,
    stats: dict[str, Any],
    final_count: int,
    renumber_from: int,
    renumbered: bool,
) -> dict[str, Any]:
    return {
        "input_file": str(input_path),
        "id_field_detected": id_field,
        "renumbered": renumbered,
        "renumbered_from": renumber_from if renumbered else None,
        "renumbered_to": renumber_from + final_count - 1 if renumbered and final_count else None,
        "final_count": final_count,
        **stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean one tile KMZ/KML/GeoJSON file and renumber field IDs from 1."
    )
    parser.add_argument("--input", required=True, type=Path, help="Single KMZ/KML/GeoJSON file")
    parser.add_argument("--output", type=Path, default=None, help="Output GeoJSON path")
    parser.add_argument(
        "--id-field",
        default="auto",
        help="ID attribute name (e.g. id_2, id_3) or 'auto' (default)",
    )
    parser.add_argument(
        "--prefer",
        choices=["first", "last", "largest"],
        default="first",
        help="When duplicate numeric IDs exist in the file (default: first)",
    )
    parser.add_argument(
        "--renumber-from",
        type=int,
        default=1,
        help="Start numbering fields from this integer (default: 1)",
    )
    parser.add_argument("--no-renumber", action="store_true", help="Skip final renumbering")
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
    parser.add_argument(
        "--simplify-tolerance-m",
        type=float,
        default=None,
        help="Simplify boundaries in UTM meters to reduce staircase vertices (e.g. 1.5)",
    )
    parser.add_argument(
        "--utm-epsg",
        type=int,
        default=32648,
        help="Projected CRS for simplification in meters (default: 32648 UTM 48N)",
    )
    parser.add_argument("--report", type=Path, default=None, help="Cleanup report JSON path")
    args = parser.parse_args()

    features = parse_vector_file(args.input)
    attr_names = set()
    for feature in features:
        attr_names.update(feature["attrs"].keys())
    id_field = detect_id_field(attr_names, args.id_field)

    cleaned, stats = cleanup_features(
        features,
        id_field=id_field,
        prefer=args.prefer,
        dedupe_spatial=args.dedupe_spatial,
        drop_added_overlapping_original=args.drop_added_overlapping_original,
        simplify_tolerance_m=args.simplify_tolerance_m,
        utm_epsg=args.utm_epsg,
    )

    if not args.no_renumber:
        rows = renumber_features(
            cleaned,
            id_field=id_field,
            renumber_from=args.renumber_from,
            keep_legacy_ids=not args.no_keep_legacy_ids,
        )
    else:
        rows = []
        for feature in cleaned:
            row = {
                "name": feature.get("name"),
                "merge_class": feature["merge_class"],
                "geometry": feature["geometry"],
            }
            if not args.no_keep_legacy_ids:
                row["legacy_id"] = feature.get("legacy_id")
                row["legacy_name"] = feature.get("legacy_name")
            row.update(feature.get("attrs", {}))
            rows.append(row)

    gdf = features_to_gdf(rows)
    output_path = infer_output_path(args.input, args.output)
    report_path = args.report or output_path.with_suffix(".clean_report.json")

    gdf.to_file(output_path, driver="GeoJSON")

    report = build_report(
        args.input,
        id_field,
        stats,
        len(gdf),
        args.renumber_from,
        renumbered=not args.no_renumber,
    )
    report["output_geojson"] = str(output_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote cleaned GeoJSON: {output_path}")
    print(f"Wrote cleanup report:  {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
