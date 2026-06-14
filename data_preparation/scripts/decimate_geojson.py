#!/usr/bin/env python3
"""Decimate dense polygon vertices in cleaned GeoJSON (corner clusters, collinear points)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from merge_tile_kmz import clean_geometry, count_polygon_vertices, simplify_geometry


def ring_coords_xy(ring) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y, *_ in ring.coords]


def collapse_consecutive_points(
    points: list[tuple[float, float]],
    tolerance_m: float,
) -> list[tuple[float, float]]:
    if len(points) < 2 or tolerance_m <= 0:
        return points

    collapsed: list[tuple[float, float]] = [points[0]]
    for point in points[1:]:
        last = collapsed[-1]
        if math.hypot(point[0] - last[0], point[1] - last[1]) >= tolerance_m:
            collapsed.append(point)

    if len(collapsed) >= 2:
        first, last = collapsed[0], collapsed[-1]
        if math.hypot(last[0] - first[0], last[1] - first[1]) < tolerance_m:
            collapsed.pop()

    return collapsed


def remove_collinear_points(
    points: list[tuple[float, float]],
    tolerance_m: float,
    closed: bool = True,
) -> list[tuple[float, float]]:
    if len(points) < 4 or tolerance_m <= 0:
        return points

    if closed:
        if points[0] != points[-1]:
            work = points + [points[0]]
        else:
            work = points[:]
        core = work[:-1]
        if len(core) < 3:
            return points

        kept: list[tuple[float, float]] = []
        n = len(core)
        for i in range(n):
            prev_pt = core[(i - 1) % n]
            curr_pt = core[i]
            next_pt = core[(i + 1) % n]
            x0, y0 = prev_pt
            x1, y1 = curr_pt
            x2, y2 = next_pt
            seg_len = math.hypot(x2 - x0, y2 - y0)
            if seg_len < 1e-9:
                continue
            cross = abs((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0))
            perp = cross / seg_len
            if perp >= tolerance_m:
                kept.append(curr_pt)

        if len(kept) < 3:
            return points
        return kept + [kept[0]]

    kept = [points[0]]
    for i in range(1, len(points) - 1):
        x0, y0 = kept[-1]
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        seg_len = math.hypot(x2 - x0, y2 - y0)
        if seg_len < 1e-9:
            continue
        cross = abs((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0))
        perp = cross / seg_len
        if perp >= tolerance_m:
            kept.append(points[i])
    kept.append(points[-1])
    return kept


def decimate_ring_utm(
    ring_coords: list[tuple[float, float]],
    cluster_tolerance_m: float,
    collinear_tolerance_m: float,
    remove_collinear: bool,
) -> list[tuple[float, float]]:
    points = ring_coords[:]
    if points and points[0] == points[-1]:
        points = points[:-1]

    if len(points) < 3:
        return ring_coords

    points = collapse_consecutive_points(points, cluster_tolerance_m)
    if remove_collinear:
        closed = points + [points[0]]
        closed = remove_collinear_points(closed, collinear_tolerance_m, closed=True)
        points = closed[:-1]

    if len(points) < 3:
        return ring_coords

    if points[0] != points[-1]:
        points.append(points[0])
    return points


def decimate_geometry_utm(
    geom: Polygon | MultiPolygon,
    cluster_tolerance_m: float,
    collinear_tolerance_m: float,
    remove_collinear: bool,
) -> Polygon | MultiPolygon | None:
    if geom.geom_type == "Polygon":
        exterior = decimate_ring_utm(
            ring_coords_xy(geom.exterior),
            cluster_tolerance_m,
            collinear_tolerance_m,
            remove_collinear,
        )
        if len(exterior) < 4:
            return None
        holes = []
        for interior in geom.interiors:
            hole = decimate_ring_utm(
                ring_coords_xy(interior),
                cluster_tolerance_m,
                collinear_tolerance_m,
                remove_collinear,
            )
            if len(hole) >= 4:
                holes.append(hole)
        return Polygon(exterior, holes)

    polygons: list[Polygon] = []
    for poly in geom.geoms:
        decimated = decimate_geometry_utm(
            poly,
            cluster_tolerance_m,
            collinear_tolerance_m,
            remove_collinear,
        )
        if decimated is not None and not decimated.is_empty:
            polygons.append(decimated)

    if not polygons:
        return None
    if len(polygons) == 1:
        return polygons[0]
    return MultiPolygon(polygons)


def decimate_geometry(
    geom: Polygon | MultiPolygon,
    cluster_tolerance_m: float,
    collinear_tolerance_m: float,
    simplify_tolerance_m: float,
    utm_epsg: int,
    remove_collinear: bool,
) -> tuple[Polygon | MultiPolygon | None, dict[str, Any]]:
    stats = {
        "vertices_before": count_polygon_vertices(geom),
        "vertices_after": count_polygon_vertices(geom),
        "changed": False,
        "dropped": False,
    }

    utm_series = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(epsg=utm_epsg)
    utm_geom = utm_series.iloc[0]
    if utm_geom is None or utm_geom.is_empty:
        stats["dropped"] = True
        return None, stats

    decimated = decimate_geometry_utm(
        utm_geom,
        cluster_tolerance_m=cluster_tolerance_m,
        collinear_tolerance_m=collinear_tolerance_m,
        remove_collinear=remove_collinear,
    )
    if decimated is None or decimated.is_empty:
        stats["dropped"] = True
        return None, stats

    wgs84_geom = gpd.GeoSeries([decimated], crs=utm_epsg).to_crs("EPSG:4326").iloc[0]
    if simplify_tolerance_m > 0:
        simplified = simplify_geometry(wgs84_geom, simplify_tolerance_m, utm_epsg=utm_epsg)
        if simplified is not None:
            wgs84_geom = simplified

    cleaned, _ = clean_geometry(wgs84_geom)
    if cleaned is None or cleaned.is_empty:
        stats["dropped"] = True
        return None, stats

    stats["vertices_after"] = count_polygon_vertices(cleaned)
    stats["changed"] = stats["vertices_after"] != stats["vertices_before"]
    return cleaned, stats


def infer_output_path(input_path: Path, output: Path | None) -> Path:
    if output is not None:
        return output
    stem = input_path.stem
    if stem.endswith("_cleaned"):
        stem = stem[: -len("_cleaned")] + "_decimated"
    else:
        stem = f"{stem}_decimated"
    return input_path.with_name(f"{stem}.geojson")


def process_file(
    input_path: Path,
    output_path: Path,
    cluster_tolerance_m: float,
    collinear_tolerance_m: float,
    simplify_tolerance_m: float,
    utm_epsg: int,
    remove_collinear: bool,
    only_class: str | None,
) -> dict[str, Any]:
    gdf = gpd.read_file(input_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    rows: list[dict[str, Any]] = []
    geometries = []
    total_before = 0
    total_after = 0
    changed_count = 0
    dropped_count = 0
    skipped_count = 0

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            dropped_count += 1
            continue

        merge_class = row.get("merge_class")
        if only_class and merge_class != only_class:
            cleaned, stats = geom, {
                "vertices_before": count_polygon_vertices(geom),
                "vertices_after": count_polygon_vertices(geom),
                "changed": False,
                "dropped": False,
            }
            skipped_count += 1
        else:
            cleaned, stats = decimate_geometry(
                geom,
                cluster_tolerance_m=cluster_tolerance_m,
                collinear_tolerance_m=collinear_tolerance_m,
                simplify_tolerance_m=simplify_tolerance_m,
                utm_epsg=utm_epsg,
                remove_collinear=remove_collinear,
            )

        if cleaned is None:
            dropped_count += 1
            continue

        total_before += stats["vertices_before"]
        total_after += stats["vertices_after"]
        if stats["changed"]:
            changed_count += 1

        out_row = {col: row[col] for col in gdf.columns if col != "geometry"}
        rows.append(out_row)
        geometries.append(cleaned)

    out_gdf = gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_gdf.to_file(output_path, driver="GeoJSON")

    return {
        "input_file": str(input_path),
        "output_geojson": str(output_path),
        "feature_count": len(out_gdf),
        "features_changed": changed_count,
        "features_skipped": skipped_count,
        "features_dropped": dropped_count,
        "vertices_before": total_before,
        "vertices_after": total_after,
        "vertices_removed": max(0, total_before - total_after),
        "cluster_tolerance_m": cluster_tolerance_m,
        "collinear_tolerance_m": collinear_tolerance_m if remove_collinear else None,
        "simplify_tolerance_m": simplify_tolerance_m if simplify_tolerance_m > 0 else None,
        "utm_epsg": utm_epsg,
        "only_class": only_class,
    }


def collect_inputs(input_path: Path | None, input_dir: Path | None) -> list[Path]:
    if input_path is not None:
        return [input_path]
    if input_dir is not None:
        return sorted(input_dir.glob("*.geojson"))
    raise SystemExit("Provide --input or --input-dir.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decimate dense vertices in cleaned GeoJSON polygons."
    )
    parser.add_argument("--input", type=Path, help="Single cleaned GeoJSON file")
    parser.add_argument("--input-dir", type=Path, help="Directory of cleaned GeoJSON files")
    parser.add_argument("--output", type=Path, help="Output GeoJSON for single --input")
    parser.add_argument("--output-dir", type=Path, help="Output directory for batch mode")
    parser.add_argument("--report", type=Path, help="Report JSON for single file")
    parser.add_argument("--report-dir", type=Path, help="Report directory for batch mode")
    parser.add_argument(
        "--cluster-tolerance-m",
        type=float,
        default=0.5,
        help="Collapse consecutive vertices within this distance in meters (default: 0.5)",
    )
    parser.add_argument(
        "--collinear-tolerance-m",
        type=float,
        default=0.25,
        help="Remove middle points within this perpendicular distance of a straight edge (default: 0.25)",
    )
    parser.add_argument(
        "--simplify-tolerance-m",
        type=float,
        default=0.75,
        help="Optional Douglas-Peucker simplify in meters after decimation (default: 0.75, 0=off)",
    )
    parser.add_argument(
        "--utm-epsg",
        type=int,
        default=32648,
        help="Projected CRS for meter-based operations (default: 32648)",
    )
    parser.add_argument(
        "--no-collinear",
        action="store_true",
        help="Skip collinear vertex removal",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip final light simplify pass",
    )
    parser.add_argument(
        "--only-class",
        choices=["original", "added"],
        default=None,
        help="Only decimate features with this merge_class value",
    )
    args = parser.parse_args()

    inputs = collect_inputs(args.input, args.input_dir)
    if not inputs:
        raise SystemExit("No GeoJSON files found.")

    simplify_tolerance_m = 0.0 if args.no_simplify else args.simplify_tolerance_m
    reports: list[dict[str, Any]] = []

    for input_path in inputs:
        if args.output_dir is not None:
            stem = input_path.stem
            if stem.endswith("_cleaned"):
                out_name = stem[: -len("_cleaned")] + "_decimated.geojson"
            else:
                out_name = f"{stem}_decimated.geojson"
            output_path = args.output_dir / out_name
        else:
            output_path = infer_output_path(input_path, args.output)

        report = process_file(
            input_path,
            output_path,
            cluster_tolerance_m=args.cluster_tolerance_m,
            collinear_tolerance_m=args.collinear_tolerance_m,
            simplify_tolerance_m=simplify_tolerance_m,
            utm_epsg=args.utm_epsg,
            remove_collinear=not args.no_collinear,
            only_class=args.only_class,
        )
        reports.append(report)

        if args.report and len(inputs) == 1:
            report_path = args.report
        elif args.report_dir is not None:
            report_path = args.report_dir / f"{output_path.stem}.decimate_report.json"
        else:
            report_path = output_path.with_suffix(".decimate_report.json")

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote: {output_path}")
        print(f"Report: {report_path}")
        print(json.dumps(report, indent=2))

    if len(reports) > 1:
        summary = {
            "files_processed": len(reports),
            "total_features": sum(r["feature_count"] for r in reports),
            "total_vertices_before": sum(r["vertices_before"] for r in reports),
            "total_vertices_after": sum(r["vertices_after"] for r in reports),
            "total_vertices_removed": sum(r["vertices_removed"] for r in reports),
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
