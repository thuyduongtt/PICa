import torch
import clip
from PIL import Image
import argparse
from pathlib import Path
import ijson

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
    for img in Path(split).iterdir():
        if img.name.startswith('.'):
            continue
        image = preprocess(Image.open(f'{ds_root_dir}/{ds_name}/{split}/{img.name}')).unsqueeze(0).to(device)
        with torch.no_grad():
            im_feat = model.encode_image(image)
            features_list.append(im_feat)
    np.save(f'{ds_root_dir}/{ds_name}/{ds_name}_{split}_feats.npy', features_list)


def extract_question_feat(ds_root_dir, ds_name, split='train'):
    json_data = stream_data(f'{ds_root_dir}/{ds_name}/{split}.json')

    features_list = []
    for d in json_data:
        if len(d['answers']) == 0:
            continue
        with torch.no_grad():
            txt_feat = model.encode_image(d['question'])
            features_list.append(txt_feat)
    np.save(f'{ds_root_dir}/{ds_name}/{ds_name}_{split}_questions_feats.npy', features_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_root_dir', type=str, require=True)
    parser.add_argument('--ds_name', type=str, require=True)
