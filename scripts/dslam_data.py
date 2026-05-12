#!/usr/bin/env python3
"""Small helper for the local Dynamic SLAM dataset registry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_registry() -> Path:
    return repo_root() / "data" / "datasets.json"


def load_registry(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dataset(registry: Dict[str, Any], dataset_id: str) -> Dict[str, Any]:
    datasets = registry.get("datasets", {})
    if dataset_id not in datasets:
        known = ", ".join(sorted(datasets))
        raise SystemExit(f"Unknown dataset id: {dataset_id}. Known ids: {known}")
    return datasets[dataset_id]


def dotted_get(obj: Dict[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def dataset_value(item: Dict[str, Any], key: str, default: Any = None) -> Any:
    if "." in key:
        return dotted_get(item, key, default)
    if key in item:
        return item[key]
    paths = item.get("paths", {})
    if key in paths:
        return paths[key]
    return default


def count_index(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                count += 1
    return count


def first_rgb_image(sequence_root: Path) -> Path:
    rgb_index = sequence_root / "rgb.txt"
    if rgb_index.exists():
        with rgb_index.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                _, rel_path = line.split(maxsplit=1)
                return sequence_root / rel_path
    rgb_dir = sequence_root / "rgb"
    images = sorted(rgb_dir.glob("*.png")) + sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.jpeg"))
    if images:
        return images[0]
    raise SystemExit(f"No RGB image found under {sequence_root}")


def cmd_list(args: argparse.Namespace) -> None:
    registry = load_registry(args.registry)
    print(f"{'dataset_id':42} {'role':30} {'frames':>7} path")
    print("-" * 120)
    for dataset_id, item in sorted(registry.get("datasets", {}).items()):
        sequence_root = item.get("paths", {}).get("sequence_root", "")
        print(
            f"{dataset_id:42} "
            f"{item.get('role', ''):30} "
            f"{int(item.get('frame_count', 0)):7d} "
            f"{sequence_root}"
        )


def cmd_get(args: argparse.Namespace) -> None:
    registry = load_registry(args.registry)
    item = dataset(registry, args.dataset_id)
    value = dataset_value(item, args.key)
    if value is None:
        if args.allow_empty:
            return
        raise SystemExit(f"Dataset {args.dataset_id} has no key: {args.key}")
    if isinstance(value, (dict, list)):
        print(json.dumps(value, indent=2, ensure_ascii=False))
    else:
        print(value)


def cmd_tool(args: argparse.Namespace) -> None:
    registry = load_registry(args.registry)
    tools = registry.get("tool_paths", {})
    value = tools.get(args.key)
    if value is None:
        known = ", ".join(sorted(tools))
        raise SystemExit(f"Unknown tool key: {args.key}. Known keys: {known}")
    print(value)


def cmd_first_image(args: argparse.Namespace) -> None:
    registry = load_registry(args.registry)
    item = dataset(registry, args.dataset_id)
    sequence_root = Path(dataset_value(item, "sequence_root"))
    print(first_rgb_image(sequence_root))


def cmd_check(args: argparse.Namespace) -> None:
    registry = load_registry(args.registry)
    failed = False
    for key, value in sorted(registry.get("tool_paths", {}).items()):
        path = Path(value)
        ok = path.exists()
        failed = failed or not ok
        print(f"{'OK' if ok else 'MISS'} tool.{key}: {path}")
    for dataset_id, item in sorted(registry.get("datasets", {}).items()):
        paths = item.get("paths", {})
        sequence_root = Path(paths.get("sequence_root", ""))
        assoc_count = count_index(sequence_root / "associations.txt")
        rgb_count = count_index(sequence_root / "rgb.txt")
        depth_count = count_index(sequence_root / "depth.txt")
        for path_key, value in sorted(paths.items()):
            path = Path(value)
            ok = path.exists()
            failed = failed or not ok
            print(f"{'OK' if ok else 'MISS'} {dataset_id}.{path_key}: {path}")
        if sequence_root.exists():
            print(
                f"INFO {dataset_id}: index_counts "
                f"associations={assoc_count} rgb={rgb_count} depth={depth_count}"
            )
    if failed:
        raise SystemExit(1)


def cmd_link(args: argparse.Namespace) -> None:
    registry = load_registry(args.registry)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    for dataset_id, item in sorted(registry.get("datasets", {}).items()):
        src = Path(item.get("paths", {}).get("sequence_root", ""))
        if not src.exists():
            print(f"MISS {dataset_id}: {src}")
            continue
        dst = output / dataset_id
        if dst.is_symlink() or dst.exists():
            if args.force:
                if dst.is_dir() and not dst.is_symlink():
                    raise SystemExit(f"Refusing to replace real directory: {dst}")
                dst.unlink()
            else:
                print(f"KEEP {dst}")
                continue
        os.symlink(src, dst)
        print(f"LINK {dst} -> {src}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=default_registry())
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registered datasets.")
    list_parser.set_defaults(func=cmd_list)

    get_parser = subparsers.add_parser("get", help="Get a dataset field.")
    get_parser.add_argument("dataset_id")
    get_parser.add_argument("key")
    get_parser.add_argument("--allow-empty", action="store_true")
    get_parser.set_defaults(func=cmd_get)

    tool_parser = subparsers.add_parser("tool", help="Get a registered tool path.")
    tool_parser.add_argument("key")
    tool_parser.set_defaults(func=cmd_tool)

    first_image_parser = subparsers.add_parser("first-image", help="Print the first RGB image path.")
    first_image_parser.add_argument("dataset_id")
    first_image_parser.set_defaults(func=cmd_first_image)

    check_parser = subparsers.add_parser("check", help="Check local paths.")
    check_parser.set_defaults(func=cmd_check)

    link_parser = subparsers.add_parser("link", help="Create local symlinks for dataset sequence roots.")
    link_parser.add_argument("--output", type=Path, default=repo_root() / "data" / "local")
    link_parser.add_argument("--force", action="store_true")
    link_parser.set_defaults(func=cmd_link)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
