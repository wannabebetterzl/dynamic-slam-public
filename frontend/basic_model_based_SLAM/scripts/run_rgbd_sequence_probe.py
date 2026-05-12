#!/usr/bin/env python3
# coding=utf-8

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np

from rflysim_slam_nav.world_sam_pipeline import WorldSamFilterPipeline


def load_index(path):
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            timestamp, rel_path = line.split(maxsplit=1)
            entries.append((float(timestamp), rel_path))
    return entries


def associate_rgb_depth(rgb_entries, depth_entries, max_diff):
    depth_times = np.array([item[0] for item in depth_entries], dtype=np.float64)
    pairs = []
    for rgb_time, rgb_path in rgb_entries:
        idx = int(np.argmin(np.abs(depth_times - rgb_time)))
        depth_time, depth_path = depth_entries[idx]
        if abs(depth_time - rgb_time) <= max_diff:
            pairs.append((rgb_time, rgb_path, depth_time, depth_path))
    return pairs


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Run a quick RGB-D sequence probe for the World-SAM dynamic filtering pipeline.')
    parser.add_argument('--sequence-root', required=True, help='Dataset sequence root containing rgb.txt and depth.txt.')
    parser.add_argument('--config', required=True, help='Pipeline config path.')
    parser.add_argument('--output-dir', required=True, help='Directory to save probe outputs.')
    parser.add_argument('--stride', type=int, default=15, help='Frame stride for probing.')
    parser.add_argument('--max-frames', type=int, default=60, help='Maximum number of frames to process.')
    parser.add_argument('--max-diff', type=float, default=0.04, help='Maximum timestamp difference for RGB-depth pairing.')
    parser.add_argument('--save-overlays', type=int, default=12, help='Maximum number of overlays to save.')
    parser.add_argument('--preset-name', default='', help='Optional preset label, e.g. fast/balanced/accurate.')
    parser.add_argument('--note', default='', help='Optional short note saved into the probe summary.')
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    rgb_entries = load_index(os.path.join(args.sequence_root, 'rgb.txt'))
    depth_entries = load_index(os.path.join(args.sequence_root, 'depth.txt'))
    pairs = associate_rgb_depth(rgb_entries, depth_entries, args.max_diff)
    sampled_pairs = pairs[:: max(1, args.stride)][: max(1, args.max_frames)]

    pipeline = WorldSamFilterPipeline(args.config)
    frame_stats = []
    overlay_dir = os.path.join(args.output_dir, 'overlays')
    ensure_dir(overlay_dir)

    for index, (rgb_time, rgb_rel, depth_time, depth_rel) in enumerate(sampled_pairs, 1):
        rgb_path = os.path.join(args.sequence_root, rgb_rel)
        depth_path = os.path.join(args.sequence_root, depth_rel)
        image = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if image is None or depth is None:
            continue

        result = pipeline.process(image, depth_mm=depth)
        stats = dict(result['stats'])
        stats['rgb_timestamp'] = rgb_time
        stats['depth_timestamp'] = depth_time
        stats['rgb_path'] = rgb_rel
        stats['depth_path'] = depth_rel
        frame_stats.append(stats)

        if index <= args.save_overlays:
            stem = f"{index:04d}_{rgb_time:.5f}"
            cv2.imwrite(os.path.join(overlay_dir, f'{stem}_overlay.jpg'), result['overlay'])
            cv2.imwrite(os.path.join(overlay_dir, f'{stem}_mask.png'), result['mask'] * 255)

    if not frame_stats:
        raise RuntimeError('No valid RGB-D frames were processed.')

    summary = {
        'sequence_root': args.sequence_root,
        'sequence_name': Path(args.sequence_root).name,
        'config_path': os.path.abspath(args.config),
        'config_name': Path(args.config).name,
        'preset_name': args.preset_name or 'custom',
        'note': args.note,
        'processed_frames': len(frame_stats),
        'stride': int(args.stride),
        'max_frames': int(args.max_frames),
        'mean_runtime_ms': float(np.mean([item['runtime_ms'] for item in frame_stats])),
        'mean_mask_ratio': float(np.mean([item['mask_ratio'] for item in frame_stats])),
        'mean_relevance': float(np.mean([item.get('mean_relevance', 0.0) for item in frame_stats])),
        'mean_detections': float(np.mean([item['detections'] for item in frame_stats])),
        'mean_filtered_detections': float(np.mean([item.get('filtered_detections', 0) for item in frame_stats])),
        'mean_foundation': float(np.mean([item.get('mean_foundation', 0.0) for item in frame_stats])),
        'mean_motion': float(np.mean([item.get('mean_motion', 0.0) for item in frame_stats])),
        'mean_tube_motion': float(np.mean([item.get('mean_tube_motion', 0.0) for item in frame_stats])),
        'mean_track_confirmation': float(np.mean([item.get('mean_track_confirmation', 0.0) for item in frame_stats])),
        'refresh_frames': int(sum(1 for item in frame_stats if item['mode'] == 'refresh')),
        'propagate_frames': int(sum(1 for item in frame_stats if item['mode'] == 'propagate')),
    }

    with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, 'frame_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(frame_stats, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, 'frame_stats.csv'), 'w', encoding='utf-8', newline='') as f:
        fieldnames = [
            'frame_index',
            'rgb_timestamp',
            'depth_timestamp',
            'detections',
            'filtered_detections',
            'mask_ratio',
            'mean_relevance',
            'mean_foundation',
            'mean_motion',
            'mean_tube_motion',
            'mean_track_confirmation',
            'runtime_ms',
            'mode',
            'rgb_path',
            'depth_path',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in frame_stats:
            row = {key: item.get(key, '') for key in fieldnames}
            writer.writerow(row)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
