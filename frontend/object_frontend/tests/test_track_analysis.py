import json
from pathlib import Path

from dynamic_object_frontend import analyze_object_tracks


def write_frame(root: Path, frame_id: int, timestamp: float, centroid):
    payload = {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "objects": [
            {
                "object_id": 1,
                "semantic_id": 11,
                "semantic_label": "person",
                "num_depth_pixels": 10,
                "centroid_camera": centroid,
            }
        ],
    }
    (root / "frames").mkdir(parents=True, exist_ok=True)
    (root / "frames" / f"{frame_id:06d}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_analyze_moving_track(tmp_path):
    write_frame(tmp_path, 0, 0.0, [0.0, 0.0, 1.0])
    write_frame(tmp_path, 1, 0.1, [0.2, 0.0, 1.0])
    write_frame(tmp_path, 2, 0.2, [0.4, 0.0, 1.0])
    summaries = analyze_object_tracks(tmp_path, min_motion_steps=1)
    assert len(summaries) == 1
    assert summaries[0].motion_state == "moving"

