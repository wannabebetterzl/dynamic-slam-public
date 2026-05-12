#!/usr/bin/env python3
# coding=utf-8

import argparse
import json
import os
import subprocess
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path


def load_registry(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def probe_remote_size(url):
    if shutil.which("curl"):
        cmd = ["curl", "-I", "-L", "-s", url]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if line.lower().startswith("content-length:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except ValueError:
                    return None
    return None


def download_file(url, dst_path):
    ensure_dir(Path(dst_path).parent)
    expected_size = probe_remote_size(url)
    if os.path.exists(dst_path):
        local_size = os.path.getsize(dst_path)
        if local_size > 0 and expected_size and local_size >= expected_size:
            print(f"[skip] archive exists: {dst_path}", flush=True)
            return
        if local_size > 0 and expected_size and local_size < expected_size:
            print(f"[resume] local archive is incomplete: {dst_path} ({local_size}/{expected_size})", flush=True)

    print(f"[download] {url}", flush=True)
    if shutil.which("curl"):
        if os.path.exists(dst_path) and os.path.getsize(dst_path) == 0:
            os.remove(dst_path)
        cmd = [
            "curl",
            "-L",
            "-C",
            "-",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "3",
            "--retry-all-errors",
            "--speed-time",
            "30",
            "--speed-limit",
            "1024",
            "--output",
            dst_path,
            url,
        ]
        subprocess.run(cmd, check=True)
    else:
        with urllib.request.urlopen(url, timeout=60) as response, open(dst_path, "wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
    print(f"[done] {dst_path}", flush=True)


def extract_archive(archive_path, extract_dir, archive_type):
    marker = os.path.join(extract_dir, ".extracted")
    if os.path.exists(marker):
        print(f"[skip] already extracted: {extract_dir}", flush=True)
        return

    ensure_dir(extract_dir)
    print(f"[extract] {archive_path} -> {extract_dir}", flush=True)
    try:
        if archive_type == "zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(extract_dir)
        elif archive_type == "tgz":
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(extract_dir)
        else:
            raise ValueError(f"Unsupported archive type: {archive_type}")
    except Exception:
        shutil.rmtree(extract_dir, ignore_errors=True)
        raise

    Path(marker).write_text("ok\n", encoding="utf-8")


def download_ground_truth(url, dst_path):
    if not url:
        return
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        print(f"[skip] ground truth exists: {dst_path}", flush=True)
        return
    print(f"[download] {url}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response, open(dst_path, "wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    print(f"[done] {dst_path}", flush=True)


def materialize_dataset(root_dir, dataset_id, cfg):
    datasets_root = os.path.join(root_dir, "datasets")
    downloads_root = os.path.join(datasets_root, "_downloads")
    ensure_dir(downloads_root)

    archive_name = os.path.basename(cfg["archive_url"])
    archive_path = os.path.join(downloads_root, archive_name)
    extract_dir = os.path.join(root_dir, cfg["extract_dir"])

    download_file(cfg["archive_url"], archive_path)
    extract_archive(archive_path, extract_dir, cfg["archive_type"])

    gt_url = cfg.get("ground_truth_url")
    if gt_url:
        gt_name = os.path.basename(gt_url)
        download_ground_truth(gt_url, os.path.join(extract_dir, gt_name))

    manifest_path = os.path.join(extract_dir, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_id": dataset_id,
                "title": cfg["title"],
                "official_page": cfg["official_page"],
                "archive_url": cfg["archive_url"],
                "ground_truth_url": cfg.get("ground_truth_url"),
                "recommended_phase": cfg.get("recommended_phase"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[manifest] {manifest_path}", flush=True)


def list_registry(registry):
    for dataset_id, cfg in registry["datasets"].items():
        print(f"{dataset_id}: {cfg['title']}")
        print(f"  group={cfg['group']}")
        print(f"  phase={cfg.get('recommended_phase', '')}")
        print(f"  archive={cfg['archive_url']}")


def resolve_dataset_ids(registry, args):
    if args.bundle:
        bundle = registry.get("bundles", {}).get(args.bundle)
        if not bundle:
            raise KeyError(f"Unknown bundle: {args.bundle}")
        return bundle
    if args.dataset_ids:
        return args.dataset_ids
    raise ValueError("Provide --bundle or --dataset-id.")


def main():
    parser = argparse.ArgumentParser(description="Download official research datasets for the multi-UAV world SLAM project.")
    parser.add_argument("--registry", default="config/research_dataset_registry.json", help="Dataset registry JSON path.")
    parser.add_argument("--root-dir", default=".", help="Project root directory.")
    parser.add_argument("--list", action="store_true", help="List registered datasets and exit.")
    parser.add_argument("--bundle", help="Download a named bundle from the registry.")
    parser.add_argument("--dataset-id", dest="dataset_ids", action="append", help="Download a specific dataset id. Repeatable.")
    args = parser.parse_args()

    registry_path = os.path.join(args.root_dir, args.registry) if not os.path.isabs(args.registry) else args.registry
    registry = load_registry(registry_path)
    if args.list:
        list_registry(registry)
        return 0

    dataset_ids = resolve_dataset_ids(registry, args)
    for dataset_id in dataset_ids:
        cfg = registry["datasets"].get(dataset_id)
        if cfg is None:
            raise KeyError(f"Unknown dataset id: {dataset_id}")
        materialize_dataset(args.root_dir, dataset_id, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
