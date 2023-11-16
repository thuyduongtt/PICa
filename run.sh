#!/bin/bash

N=2

case $1 in
  1)
    DS_NAME="unbalanced"
    ;;
  2)
    DS_NAME="balanced_10"
    ;;
esac

python extract_features.py --ds_root_dir data/datasets --ds_name $DS_NAME

