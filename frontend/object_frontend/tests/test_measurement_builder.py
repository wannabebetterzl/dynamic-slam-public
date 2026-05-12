import numpy as np

from dynamic_object_frontend import CameraIntrinsics, DetectionRecord, build_object_observations


def test_build_object_observation_from_depth_mask():
    depth = np.full((8, 8), 5000, dtype=np.uint16)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    detections = [
        DetectionRecord(
            object_id=1,
            semantic_id=11,
            semantic_label="person",
            bbox_2d=(1, 1, 6, 6),
            dynamic_score=0.8,
            temporal_consistency=0.9,
            geometry_dynamic_score=0.2,
            filter_out=True,
        )
    ]
    observations, clouds = build_object_observations(
        frame_id=0,
        timestamp=1.0,
        depth=depth,
        binary_mask=mask,
        detections=detections,
        intrinsics=CameraIntrinsics(fx=4.0, fy=4.0, cx=4.0, cy=4.0),
    )
    assert len(observations) == 1
    assert observations[0].num_depth_pixels == 16
    assert observations[0].centroid_camera is not None
    assert clouds[1].shape == (16, 3)


def test_instance_mask_overrides_binary_bbox_assignment():
    depth = np.full((6, 6), 5000, dtype=np.uint16)
    binary_mask = np.zeros((6, 6), dtype=np.uint8)
    binary_mask[1:5, 1:5] = 255
    instance_mask = np.zeros((6, 6), dtype=np.uint16)
    instance_mask[2:4, 2:4] = 3
    detections = [
        DetectionRecord(
            object_id=3,
            semantic_id=11,
            semantic_label="person",
            bbox_2d=(1, 1, 4, 4),
            dynamic_score=0.8,
            temporal_consistency=0.9,
            geometry_dynamic_score=0.2,
            filter_out=True,
        )
    ]
    observations, clouds = build_object_observations(
        frame_id=0,
        timestamp=1.0,
        depth=depth,
        binary_mask=binary_mask,
        instance_mask=instance_mask,
        detections=detections,
        intrinsics=CameraIntrinsics(fx=4.0, fy=4.0, cx=4.0, cy=4.0),
    )
    assert observations[0].num_mask_pixels == 4
    assert clouds[3].shape == (4, 3)


def test_object_observation_preserves_association_scores():
    depth = np.full((4, 4), 5000, dtype=np.uint16)
    mask = np.ones((4, 4), dtype=np.uint8) * 255
    detections = [
        DetectionRecord(
            object_id=5,
            semantic_id=11,
            semantic_label="person",
            bbox_2d=(0, 0, 3, 3),
            dynamic_score=0.8,
            temporal_consistency=0.9,
            geometry_dynamic_score=0.2,
            filter_out=True,
            match_score=0.77,
            association_bbox_iou=0.61,
            association_mask_iou=0.52,
            association_appearance=0.43,
            association_depth=0.34,
            association_id_match=1.0,
            temporal_fusion_score=0.81,
            temporal_id_consistency=0.72,
            temporal_mask_agreement=0.63,
            temporal_box_agreement=0.54,
        )
    ]
    observations, _clouds = build_object_observations(
        frame_id=0,
        timestamp=1.0,
        depth=depth,
        binary_mask=mask,
        detections=detections,
        intrinsics=CameraIntrinsics(fx=4.0, fy=4.0, cx=2.0, cy=2.0),
    )

    obs = observations[0]
    assert obs.match_score == 0.77
    assert obs.association_bbox_iou == 0.61
    assert obs.association_mask_iou == 0.52
    assert obs.association_appearance == 0.43
    assert obs.association_depth == 0.34
    assert obs.association_id_match == 1.0
    assert obs.temporal_fusion_score == 0.81
    assert obs.temporal_id_consistency == 0.72
    assert obs.temporal_mask_agreement == 0.63
    assert obs.temporal_box_agreement == 0.54
