import json
from pathlib import Path

import cv2
import numpy as np

from dynamic_object_frontend import build_dynosam_adapter_bundle, export_dynosam_bundle


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_direct_dynosam_frame_packet_and_export(tmp_path):
    sequence_root = tmp_path / "sequence"
    packet_root = tmp_path / "packet"
    out_root = tmp_path / "bundle_out"
    sequence_root.mkdir(parents=True, exist_ok=True)
    packet_root.mkdir(parents=True, exist_ok=True)

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[1:3, 1:3] = 255
    depth = np.full((4, 4), 5000, dtype=np.uint16)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255

    cv2.imwrite(str(sequence_root / "rgb.png"), rgb)
    cv2.imwrite(str(sequence_root / "filtered_depth.png"), depth)
    cv2.imwrite(str(sequence_root / "raw_depth.png"), depth)
    cv2.imwrite(str(sequence_root / "dynamic_mask.png"), mask)
    instance_mask = np.zeros((4, 4), dtype=np.uint16)
    instance_mask[0, 0] = 7
    cv2.imwrite(str(sequence_root / "instance_mask.png"), instance_mask)

    _write_json(
        packet_root / "manifest.json",
        {
            "sequence_root": str(sequence_root),
            "pose_convention": "camera_pose_twc maps camera-frame points into world frame",
        },
    )
    _write_json(
        packet_root / "frames.json",
        [
            {
                "frame_id": 0,
                "timestamp": 1.25,
                "rgb": "rgb.png",
                "filtered_depth": "filtered_depth.png",
                "raw_depth": "raw_depth.png",
                "mask": "dynamic_mask.png",
                "instance_mask": "instance_mask.png",
                "camera_pose_twc": {
                    "timestamp": 1.25,
                    "translation": [0.0, 0.0, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        ],
    )
    _write_json(
        packet_root / "object_observations.json",
        [
            {
                "frame_id": 0,
                "timestamp": 1.25,
                "object_id": 7,
                "semantic_id": 11,
                "semantic_label": "person",
                "bbox_2d": [1, 1, 2, 2],
                "num_depth_pixels": 4,
                "centroid_camera": [0.0, 0.0, 1.0],
                "confidence": 0.9,
                "dynamic_score": 0.9,
                "temporal_consistency": 0.8,
            }
        ],
    )
    _write_json(packet_root / "tracks.json", [{"object_id": 7, "frames": 1}])

    adapter_bundle = build_dynosam_adapter_bundle(packet_root=packet_root)
    assert adapter_bundle.validation["frames"] == 1
    assert adapter_bundle.validation["objects"] == 1

    direct_packet = next(adapter_bundle.iter_direct_frame_packets())
    assert direct_packet.raw_rgb.shape == (4, 4, 3)
    assert direct_packet.static_filtered_depth_metric.shape == (4, 4)
    assert float(direct_packet.static_filtered_depth_metric[0, 0]) == 1.0
    assert direct_packet.instance_mask[0, 0] == 7
    assert int((direct_packet.instance_mask > 0).sum()) == 1

    manifest = export_dynosam_bundle(packet_root=packet_root, output_root=out_root, materialize_mode="copy")
    assert manifest["validation"]["frames"] == 1
    assert (out_root / "camera_info.json").is_file()
    assert (out_root / "frames.json").is_file()
    assert (out_root / "instance_masks" / "000000_1.250000_instances.npy").is_file()


def test_adapter_keeps_raw_rgb_separate_from_filtered_rgb(tmp_path):
    source_sequence = tmp_path / "source_sequence"
    sequence_root = tmp_path / "sequence"
    packet_root = tmp_path / "packet"
    out_root = tmp_path / "bundle_out"
    source_sequence.mkdir(parents=True, exist_ok=True)
    sequence_root.mkdir(parents=True, exist_ok=True)
    packet_root.mkdir(parents=True, exist_ok=True)

    raw_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    raw_rgb[:, :] = (7, 11, 13)
    filtered_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    filtered_rgb[:, :] = (101, 103, 107)
    depth = np.full((4, 4), 5000, dtype=np.uint16)
    mask = np.zeros((4, 4), dtype=np.uint8)
    instance_mask = np.zeros((4, 4), dtype=np.uint16)
    instance_mask[1:3, 1:3] = 9

    cv2.imwrite(str(source_sequence / "raw_rgb.png"), raw_rgb)
    cv2.imwrite(str(sequence_root / "filtered_rgb.png"), filtered_rgb)
    cv2.imwrite(str(sequence_root / "filtered_depth.png"), depth)
    cv2.imwrite(str(sequence_root / "raw_depth.png"), depth)
    cv2.imwrite(str(sequence_root / "dynamic_mask.png"), mask)
    cv2.imwrite(str(sequence_root / "instance_mask.png"), instance_mask)

    _write_json(
        packet_root / "manifest.json",
        {
            "sequence_root": str(sequence_root),
            "pose_convention": "camera_pose_twc maps camera-frame points into world frame",
        },
    )
    _write_json(
        packet_root / "frames.json",
        [
            {
                "frame_id": 0,
                "timestamp": 2.5,
                "rgb": "filtered_rgb.png",
                "filtered_depth": "filtered_depth.png",
                "raw_depth": "raw_depth.png",
                "mask": "dynamic_mask.png",
                "instance_mask": "instance_mask.png",
                "camera_pose_twc": None,
            }
        ],
    )
    _write_json(packet_root / "object_observations.json", [])
    _write_json(packet_root / "tracks.json", [])
    frame_stats = tmp_path / "frame_stats.json"
    benchmark_summary = tmp_path / "benchmark_summary.json"
    _write_json(frame_stats, [{"frame_index": 1, "rgb_path": "raw_rgb.png"}])
    _write_json(benchmark_summary, {"sequence_root": str(source_sequence)})

    adapter_bundle = build_dynosam_adapter_bundle(
        packet_root=packet_root,
        benchmark_summary_path=benchmark_summary,
        frame_stats_path=frame_stats,
    )
    packet = next(adapter_bundle.iter_direct_frame_packets())
    assert tuple(int(v) for v in packet.raw_rgb[0, 0]) == (7, 11, 13)
    assert tuple(int(v) for v in packet.static_filtered_rgb[0, 0]) == (101, 103, 107)

    manifest = export_dynosam_bundle(
        packet_root=packet_root,
        output_root=out_root,
        benchmark_summary_path=benchmark_summary,
        frame_stats_path=frame_stats,
        materialize_mode="copy",
    )
    camera_info = json.loads((out_root / "camera_info.json").read_text())
    frames = json.loads((out_root / "frames.json").read_text())
    exported_rgb = cv2.imread(str(out_root / frames[0]["raw_rgb"]), cv2.IMREAD_COLOR)

    assert manifest["data_contract"]["dynosam_frontend_rgb"] == "raw_rgb"
    assert camera_info["image_source"] == "raw_rgb"
    assert tuple(int(v) for v in exported_rgb[0, 0]) == (7, 11, 13)


def test_export_dynosam_bundle_can_materialize_forward_optical_flow(tmp_path):
    sequence_root = tmp_path / "sequence"
    packet_root = tmp_path / "packet"
    out_root = tmp_path / "bundle_out"
    sequence_root.mkdir(parents=True, exist_ok=True)
    packet_root.mkdir(parents=True, exist_ok=True)

    depth = np.full((16, 16), 5000, dtype=np.uint16)
    mask = np.zeros((16, 16), dtype=np.uint8)
    instance_mask = np.zeros((16, 16), dtype=np.uint16)
    for idx, x0 in enumerate([3, 5]):
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[5:10, x0 : x0 + 5] = 255
        cv2.imwrite(str(sequence_root / f"rgb_{idx}.png"), rgb)
        cv2.imwrite(str(sequence_root / f"depth_{idx}.png"), depth)
        cv2.imwrite(str(sequence_root / f"mask_{idx}.png"), mask)
        cv2.imwrite(str(sequence_root / f"instance_{idx}.png"), instance_mask)

    _write_json(
        packet_root / "manifest.json",
        {
            "sequence_root": str(sequence_root),
            "pose_convention": "camera_pose_twc maps camera-frame points into world frame",
        },
    )
    _write_json(
        packet_root / "frames.json",
        [
            {
                "frame_id": idx,
                "timestamp": float(idx),
                "rgb": f"rgb_{idx}.png",
                "filtered_depth": f"depth_{idx}.png",
                "raw_depth": f"depth_{idx}.png",
                "mask": f"mask_{idx}.png",
                "instance_mask": f"instance_{idx}.png",
                "camera_pose_twc": None,
            }
            for idx in range(2)
        ],
    )
    _write_json(packet_root / "object_observations.json", [])
    _write_json(packet_root / "tracks.json", [])

    manifest = export_dynosam_bundle(
        packet_root=packet_root,
        output_root=out_root,
        materialize_mode="copy",
        optical_flow_method="farneback",
    )
    frames = json.loads((out_root / "frames.json").read_text())
    flow = np.load(out_root / frames[0]["optical_flow_npy"])

    assert manifest["validation"]["optical_flow_method"] == "farneback"
    assert manifest["validation"]["optical_flow_missing"] == 0
    assert manifest["data_contract"]["dynosam_frontend_optical_flow"] == "optical_flow_npy_forward_float32_optional"
    assert flow.shape == (16, 16, 2)
    assert flow.dtype == np.float32
