from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window, bounds as window_bounds, transform as window_transform
from shapely.geometry import MultiPolygon, Polygon, box
from PIL import Image


def load_pairs_csv(pairs_csv: Path) -> list[dict[str, str]]:
    with pairs_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def chip_offsets(image_size: int, chip_size: int, stride: int) -> list[int]:
    if image_size < chip_size:
        return []
    offsets = list(range(0, image_size - chip_size + 1, stride))
    last = image_size - chip_size
    if not offsets or offsets[-1] != last:
        offsets.append(last)
    return offsets


def flatten_polygons(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if not g.is_empty and isinstance(g, Polygon)]
    if geom.geom_type == "GeometryCollection":
        polys: list[Polygon] = []
        for part in geom.geoms:
            polys.extend(flatten_polygons(part))
        return polys
    return []


def polygon_to_yolo_line(
    polygon: Polygon,
    inv_transform,
    chip_size: int,
    class_id: int = 0,
    coord_digits: int = 6,
) -> str | None:
    if polygon.is_empty or polygon.area <= 0:
        return None

    coords = list(polygon.exterior.coords)
    if coords and coords[0] == coords[-1]:
        coords = coords[:-1]
    if len(coords) < 3:
        return None

    normalized: list[float] = []
    for x_map, y_map in coords:
        col, row = inv_transform * (x_map, y_map)
        x_norm = max(0.0, min(1.0, col / chip_size))
        y_norm = max(0.0, min(1.0, row / chip_size))
        normalized.extend([x_norm, y_norm])

    if len(normalized) < 6:
        return None

    parts = " ".join(f"{v:.{coord_digits}f}" for v in normalized)
    return f"{class_id} {parts}"


def write_data_yaml(output_dir: Path) -> None:
    yaml_path = output_dir / "data.yaml"
    content = f"""path: {output_dir.resolve()}
        train: images/train
        val: images/val

        nc: 1
        names: ['field']
        """
    yaml_path.write_text(content, encoding="utf-8")


def chip_tile_pair(
    tif_path: Path,
    geojson_path: Path,
    output_dir: Path,
    split: str,
    tile_id: str,
    chip_size: int = 512,
    stride: int = 256,
    class_id: int = 0,
    min_area_m2: float = 25.0,
    max_chips: int | None = None,
) -> dict[str, Any]:
    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    stats: dict[str, Any] = {
        "tile_id": tile_id,
        "split": split,
        "tif_path": str(tif_path),
        "geojson_path": str(geojson_path),
        "chip_size": chip_size,
        "stride": stride,
        "chips_written": 0,
        "chips_with_labels": 0,
        "chips_empty": 0,
        "label_instances": 0,
        "instances_skipped_small": 0,
    }

    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {tif_path}")

        gdf = gdf.to_crs(src.crs)
        spatial_index = gdf.sindex if len(gdf) > 0 else None

        band_count = src.count
        read_indexes = (1, 2, 3) if band_count >= 3 else (1,)

        row_offsets = chip_offsets(src.height, chip_size, stride)
        col_offsets = chip_offsets(src.width, chip_size, stride)
        chip_index = 0

        for row_off in row_offsets:
            for col_off in col_offsets:
                if max_chips is not None and stats["chips_written"] >= max_chips:
                    return stats

                window = Window(col_off, row_off, chip_size, chip_size)
                data = src.read(indexes=read_indexes, window=window, boundless=False)

                if band_count >= 3:
                    rgb = np.transpose(data[:3], (1, 2, 0))
                else:
                    gray = data[0]
                    rgb = np.stack([gray, gray, gray], axis=-1)

                if rgb.dtype != np.uint8:
                    if rgb.max() <= 1.0:
                        rgb = (rgb * 255).astype(np.uint8)
                    else:
                        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

                chip_name = f"{tile_id}_chip{chip_index:04d}"
                chip_index += 1
                image_path = images_dir / f"{chip_name}.png"
                label_path = labels_dir / f"{chip_name}.txt"

                Image.fromarray(rgb).save(image_path)

                win_transform = window_transform(window, src.transform)
                inv_transform = ~win_transform
                chip_bounds = window_bounds(window, src.transform)
                chip_box = box(*chip_bounds)

                yolo_lines: list[str] = []
                if spatial_index is not None:
                    candidate_idx = list(spatial_index.intersection(chip_bounds))
                    for idx in candidate_idx:
                        geom = gdf.geometry.iloc[idx]
                        if geom is None or geom.is_empty or not geom.intersects(chip_box):
                            continue

                        clipped = geom.intersection(chip_box)
                        for poly in flatten_polygons(clipped):
                            if poly.area < min_area_m2:
                                stats["instances_skipped_small"] += 1
                                continue
                            line = polygon_to_yolo_line(
                                poly,
                                inv_transform,
                                chip_size,
                                class_id=class_id,
                            )
                            if line:
                                yolo_lines.append(line)

                label_path.write_text(
                    ("\n".join(yolo_lines) + "\n") if yolo_lines else "",
                    encoding="utf-8",
                )

                stats["chips_written"] += 1
                if yolo_lines:
                    stats["chips_with_labels"] += 1
                    stats["label_instances"] += len(yolo_lines)
                else:
                    stats["chips_empty"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chip tile imagery and build YOLO segmentation labels."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="data_preparation directory",
    )
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=None,
        help="tile_pairs.csv path (default: datasets/tiles/tile_pairs.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dataset output root (default: datasets/cambodia-fields)",
    )
    parser.add_argument(
        "--tile-id",
        action="append",
        default=None,
        help="Process only this tile_id (repeatable). Default: all rows in CSV.",
    )
    parser.add_argument("--chip-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument(
        "--min-area-m2",
        type=float,
        default=25.0,
        help="Skip clipped polygons smaller than this area in map units squared",
    )
    parser.add_argument(
        "--max-chips",
        type=int,
        default=None,
        help="Limit chips per tile (useful for pilot runs)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for per-tile JSON reports",
    )
    parser.add_argument(
        "--write-data-yaml",
        action="store_true",
        help="Write data.yaml in output-dir",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    pairs_csv = args.pairs_csv or (base_dir / "datasets" / "tiles" / "tile_pairs.csv")
    output_dir = args.output_dir or (base_dir / "datasets" / "cambodia-fields")
    report_dir = args.report_dir or (output_dir / "reports")

    if not pairs_csv.is_file():
        raise SystemExit(f"Pairs CSV not found: {pairs_csv}")

    pairs = load_pairs_csv(pairs_csv)
    if args.tile_id:
        allowed = set(args.tile_id)
        pairs = [row for row in pairs if row["tile_id"] in allowed]
        if not pairs:
            raise SystemExit(f"No matching tile_id in CSV: {sorted(allowed)}")

    reports: list[dict[str, Any]] = []
    for row in pairs:
        tile_id = row["tile_id"]
        split = row["split"]
        tif_path = base_dir / row["tif_path"]
        geojson_path = base_dir / row["geojson_path"]

        if not tif_path.is_file():
            raise SystemExit(f"TIF not found: {tif_path}")
        if not geojson_path.is_file():
            raise SystemExit(f"GeoJSON not found: {geojson_path}")

        print(f"Chipping {tile_id} ({split})...")
        report = chip_tile_pair(
            tif_path=tif_path,
            geojson_path=geojson_path,
            output_dir=output_dir,
            split=split,
            tile_id=tile_id,
            chip_size=args.chip_size,
            stride=args.stride,
            class_id=args.class_id,
            min_area_m2=args.min_area_m2,
            max_chips=args.max_chips,
        )
        reports.append(report)

        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{tile_id}.chip_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))

    if args.write_data_yaml:
        write_data_yaml(output_dir)
        print(f"Wrote {output_dir / 'data.yaml'}")

    summary = {
        "tiles_processed": len(reports),
        "total_chips": sum(r["chips_written"] for r in reports),
        "total_chips_with_labels": sum(r["chips_with_labels"] for r in reports),
        "total_label_instances": sum(r["label_instances"] for r in reports),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
