#!/bin/bash

# base: chairs only, 1024 points, 32 workers
python precompute_pointcloud.py \
  --data-dir /ist/ist-share/scads/ploy/scene2/big_file/shapenet/shapenet \
  --object-class 03001627 \
  --num-points 1024 \
  --output-dir ./sampled_poincloud \
  --workers 32

# more points: chairs only, 5000 points
python precompute_pointcloud.py \
  --data-dir /ist/ist-share/scads/ploy/scene2/big_file/shapenet/shapenet \
  --object-class 03001627 \
  --num-points 5000 \
  --output-dir ./sampled_poincloud_5000 \
  --workers 32

# more points: chairs only, 10k points
python precompute_pointcloud.py \
  --data-dir /ist/ist-share/scads/ploy/scene2/big_file/shapenet/shapenet \
  --object-class 03001627 \
  --num-points 10000 \
  --output-dir ./sampled_poincloud_10000 \
  --workers 32

# multi classes: chair, bookshelf, bed, monitor, sofa, 1024 points
python precompute_pointcloud.py \
  --data-dir /ist/ist-share/scads/ploy/scene2/big_file/shapenet/shapenet \
  --object-classes 03001627 02871439 02818832 03211117 04256520 \
  --num-points 1024 \
  --output-dir ./sampled_poincloud_5classes_1024 \
  --workers 32