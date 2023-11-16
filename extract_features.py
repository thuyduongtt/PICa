import argparse
import clip
import ijson
import numpy as np
import torch
from PIL import Image
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)


def stream_data(path_to_json_file, limit=0, start_at=0):
    i = 0
    with open(path_to_json_file) as f:
        datareader = ijson.items(f, 'questions.item')
        for record in datareader:
            i += 1
            if i < start_at + 1:
                continue
            if 0 < limit < i - start_at:
                return

            yield record


def extract_img_feat(ds_root_dir, ds_name, split='train'):
    features_list = []
    count = 0
    for img in Path(f'{ds_root_dir}/{ds_name}/{split}').iterdir():
        if img.name.startswith('.'):
            continue
        image = preprocess(Image.open(f'{ds_root_dir}/{ds_name}/{split}/{img.name}')).unsqueeze(0).to(device)
        with torch.no_grad():
            im_feat = model.encode_image(image)
            features_list.append(im_feat.cpu())
        count += 1
        if count % 100 == 0:
            print(f'[{count}] {img.name}')
    np.save(f'{ds_root_dir}/{ds_name}/{ds_name}_{split}_feats.npy', features_list)


def extract_question_feat(ds_root_dir, ds_name, split='train'):
    json_data = stream_data(f'{ds_root_dir}/{ds_name}/{split}.json')

    features_list = []
    count = 0
    for d in json_data:
        if len(d['answers']) == 0:
            continue
        txt = clip.tokenize(d['question']).to(device)
        with torch.no_grad():
            txt_feat = model.encode_text(txt)
            features_list.append(txt_feat.cpu())
        count += 1
        if count % 100 == 0:
            print(f'[{count}]')
    np.save(f'{ds_root_dir}/{ds_name}/{ds_name}_{split}_questions_feats.npy', features_list)


# def extract_image_question_idx(ds_root_dir, ds_name, split='train'):



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)  # in ['question', 'image']
    parser.add_argument('--ds_root_dir', type=str, required=True)
    parser.add_argument('--ds_name', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    args = parser.parse_args()

    if args.task == 'question':
        extract_question_feat(args.ds_root_dir, args.ds_name, args.split)
    else:
        extract_img_feat(args.ds_root_dir, args.ds_name, args.split)
