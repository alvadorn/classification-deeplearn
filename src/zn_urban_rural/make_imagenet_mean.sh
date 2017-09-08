#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=$DATASET/zn_urban_rural
DATA=$DATASET/zn_urban_rural/pos_script
TOOLS=$CAFFE/build/tools

$TOOLS/compute_image_mean $EXAMPLE/zn_urban_rural_train_lmdb \
  $DATA/zn_urban_rural.binaryproto

echo "Done."
