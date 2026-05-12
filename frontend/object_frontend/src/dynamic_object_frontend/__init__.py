"""Foundation-model object observation builders for Dynamic SLAM."""

from .types import CameraIntrinsics, DetectionRecord, ObjectObservation
from .measurement_builder import build_object_observations
from .track_analysis import ObjectTrackSummary, analyze_object_tracks

try:
    from .dynosam_adapter import (
        DynosamAdapterBundle,
        DynosamDirectFramePacket,
        DynosamFrameAdapterRecord,
        build_dynosam_adapter_bundle,
        export_dynosam_bundle,
        materialize_dynosam_bundle,
    )
except ModuleNotFoundError as exc:
    if exc.name != "cv2":
        raise

__all__ = [
    "CameraIntrinsics",
    "DetectionRecord",
    "ObjectObservation",
    "build_object_observations",
    "ObjectTrackSummary",
    "analyze_object_tracks",
]

if "build_dynosam_adapter_bundle" in globals():
    __all__.extend(
        [
            "DynosamAdapterBundle",
            "DynosamDirectFramePacket",
            "DynosamFrameAdapterRecord",
            "build_dynosam_adapter_bundle",
            "export_dynosam_bundle",
            "materialize_dynosam_bundle",
        ]
    )
