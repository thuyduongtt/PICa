#!/bin/bash

N=2

case $1 in
  1)
    TASK='question'
    DS_NAME="unbalanced"
    ;;
  2)
    TASK='question'
    DS_NAME="balanced_10"
    ;;
  3)
    TASK='image'
    DS_NAME="unbalanced"
    ;;
  4)
    TASK='image'
    DS_NAME="balanced_10"
    ;;
  3)
    TASK='idx'
    DS_NAME="unbalanced"
    ;;
  4)
    TASK='idx'
    DS_NAME="balanced_10"
    ;;
esac

python extract_features.py --task $TASK --ds_root_dir data/datasets --ds_name $DS_NAME --split train
python extract_features.py --task $TASK --ds_root_dir data/datasets --ds_name $DS_NAME --split test

