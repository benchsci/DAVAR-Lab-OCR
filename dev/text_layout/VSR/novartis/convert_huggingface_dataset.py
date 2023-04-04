#!/usr/bin/env python3
from datasets import load_dataset
from pathlib import Path
import json
from tqdm import tqdm
import argparse


SPLIT_DIR = Path('/home/jupyter/kweston_persistent_workspace/datasets/Doclynet_5_percent_annotations/')
TRAIN_SPLIT_PATH = SPLIT_DIR / 'train_random_sampled_updated_merged_boxes.json'
VAL_SPLIT_PATH = SPLIT_DIR / 'val_random_sampled_updated_merged_boxes.json'
DOCLAYNET_EXTRAS_DIR = Path('/home/jupyter/kweston_persistent_workspace/datasets/DocLayNet_extra/')

def rescale_bbox(bbox, img_size):
    x0, y0, w, h = bbox
    imw, imh = img_size
    x0 = x0 / imw
    y0 = y0 / imh
    w = w / imw
    h = h / imh
    return [x0, y0, w, h]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=Path, help='Path to output directory')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()

    with open(TRAIN_SPLIT_PATH) as f:
        train_split = json.load(f)
    with open(VAL_SPLIT_PATH) as f:
        val_split = json.load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_dataset("ds4sd/DocLayNet", split="train")

    doc_names = train_dataset['doc_name']
    page_numbers = train_dataset['page_no']
    doc_index = [(d, p) for d, p in zip(doc_names, page_numbers)]

    class_names_list = train_dataset.features['objects'][0]['category_id'].names
    with open('class_names.json', 'w') as f:
        json.dump(class_names_list, f, indent=4)

    # Collect list of pdf files and page numbers to process as well as a list of annotations
    debug_dir = args.output_dir / 'debug'
    if args.debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    pdf_page_list = []
    annotations = {}
    for _, rec in enumerate(tqdm(train_split['images'])):
        fname = str(rec['file_name']).replace('png', 'json')
        with open(DOCLAYNET_EXTRAS_DIR / 'JSON' / fname) as f:
            extras = json.load(f)
        metadata = extras['metadata']
        pdf_page_list.append((
            DOCLAYNET_EXTRAS_DIR / 'PDF' / fname.replace('json', 'pdf'),
            metadata['page_no']
        ))
        i = doc_index.index((metadata['original_filename'], metadata['page_no']))
        tmp = train_dataset[i].copy()
        pil_image = tmp['image']
        if args.debug:
            pil_image.save(debug_dir / f'{Path(fname).stem}.png')
        del tmp['image']
        # bboxes = [rescale_bbox(b, tmp['width'], tmp['height']) for b in tmp['objects']['bbox']]
        annotations[Path(fname).stem] = tmp

    # Write out annotations and pdf page list
    with open(args.output_dir / 'annotations.json', 'w') as f:
        json.dump(annotations, f, indent=4)

    with open(args.output_dir / 'pdf_page_list.csv', 'w') as f:
        f.write('pdf_path,page_number\n')
        for pdf_path, page_number in pdf_page_list:
            f.write(f'{pdf_path},{page_number}\n')

