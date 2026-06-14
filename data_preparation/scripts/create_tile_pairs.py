from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path

BRANCHES = {
    "batheay": "bth",
    "koh_andet": "tko",
    "preah_sdach": "pvg",
}
DEFAULT_VAL_TILES = {"bth_id_15", "tko_id_14", "pvg_id_15"}


def tile_key(prefix: str, tile_id: int) -> str:
    return f"{prefix}_id_{tile_id}"


def default_tiles_dir(base_dir: Path) -> Path:
    return base_dir / "datasets" / "tiles"


def default_geojson_dir(base_dir: Path) -> Path:
    return base_dir / "datasets" / "polygon_files_cleaned" / "geojson_decimated"


def find_geojson(dec_dir: Path, prefix: str, tile_id: int) -> Path | None:
    patterns = [
        f"fields_extracted_{prefix}_id_{tile_id}_decimated.geojson",
        f"field_extracted_{prefix}_id_{tile_id}_decimated.geojson",
        f"field_extracted_{prefix}_id_{tile_id}_*_decimated.geojson",
        f"fields_extracted_{prefix}_id_{tile_id}_*_decimated.geojson",
    ]
    for pattern in patterns:
        matches = sorted(dec_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def collect_pairs(
    base_dir: Path,
    tiles_dir: Path,
    geojson_dir: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for branch_folder, prefix in BRANCHES.items():
        branch_path = tiles_dir / branch_folder
        if not branch_path.exists():
            continue

        for tif in sorted(branch_path.glob("*_image_georeferenced.tif")):
            m = re.search(rf"{prefix}_id_(\d+)", tif.name)
            if not m:
                continue

            tile_id = int(m.group(1))
            key = tile_key(prefix, tile_id)
            geojson = find_geojson(geojson_dir, prefix, tile_id)
            if geojson is None:
                continue

            rows.append(
                {
                    "tile_id": key,
                    "branch": branch_folder,
                    "tif_path": str(tif.relative_to(base_dir)),
                    "geojson_path": str(geojson.relative_to(base_dir)),
                    "split": "val" if key in DEFAULT_VAL_TILES else "train",
                }
            )

    rows.sort(key=lambda r: (r["branch"], r["tile_id"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tile_pairs.csv")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="data_preparation directory",
    )
    parser.add_argument(
        "--tiles-dir",
        type=Path,
        default=None,
        help="Georeferenced TIF directory (default: datasets/tiles)",
    )
    parser.add_argument(
        "--geojson-dir",
        type=Path,
        default=None,
        help="Decimated GeoJSON directory (default: datasets/polygon_files_cleaned/geojson_decimated)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: datasets/tiles/tile_pairs.csv)",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    tiles_dir = (args.tiles_dir or default_tiles_dir(base_dir)).resolve()
    geojson_dir = (args.geojson_dir or default_geojson_dir(base_dir)).resolve()
    output = args.output or (tiles_dir / "tile_pairs.csv")

    if not tiles_dir.is_dir():
        raise SystemExit(f"Tiles directory not found: {tiles_dir}")
    if not geojson_dir.is_dir():
        raise SystemExit(f"GeoJSON directory not found: {geojson_dir}")

    rows = collect_pairs(base_dir, tiles_dir, geojson_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tile_id", "branch", "tif_path", "geojson_path", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} pairs to {output}")


if __name__ == "__main__":
    main()
