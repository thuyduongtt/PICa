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
  5)
    TASK='idx'
    DS_NAME="unbalanced"
    ;;
  6)
    TASK='idx'
    DS_NAME="balanced_10"
    ;;
  7)
    N_ENSEMBLE=1
    DS_NAME="unbalanced"
    ;;
  8)
    N_ENSEMBLE=1
    DS_NAME="balanced_10"
    ;;
  9)
    N_ENSEMBLE=5
    DS_NAME="unbalanced"
    ;;
  10)
    N_ENSEMBLE=5
    DS_NAME="balanced_10"
    ;;
esac

#python extract_features.py --task $TASK --ds_root_dir data/datasets --ds_name $DS_NAME --split train
#python extract_features.py --task $TASK --ds_root_dir data/datasets --ds_name $DS_NAME --split test

python gpt3_api_okvqa.py \
 --ds_name $DS_NAME \
 --n_ensemble $N_ENSEMBLE \
 --valcaption_file data/assets/captions_${DS_NAME}.json \
 --similarity_path data/datasets/${DS_NAME} \
 --output_path output_${N_ENSEMBLE}_${DS_NAME}
