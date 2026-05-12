# Object Front-End

This module converts YOLO-World + SAM-style front-end outputs into
factor-graph-ready object observations.

The first supported input is the copied `basic_frontend` sequence format:

```text
sequence/
  rgb/
  depth/
  mask/
  meta/
  associations.txt
  groundtruth.txt
```

`meta/*.txt` rows are expected to follow:

```text
track_id x1 y1 x2 y2 dynamic_score temporal_consistency geometry_dynamic_score filter_out
```

The converter produces per-frame JSON and NPZ files containing:

- object ID
- semantic label
- bbox
- dynamic/confidence scores
- centroid in camera coordinates
- 3D axis-aligned bounding box in camera coordinates
- sampled object point cloud

