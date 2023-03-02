import os
import json
import cv2
import copy
import glob
import multiprocessing
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path
import sys
import shutil

dirname = os.path.join(os.path.dirname(__file__), '../../../demo/text_layout/datalist/DocBank/Scripts/DocBankLoader-master')
sys.path.append(dirname)

from docbank_loader import DocBankLoader, DocBankConverter

# txt_dir = 'demo/txt'
# img_dir = 'demo/img'

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=Path, help='The output directory of pdf_converter.py containing box orig and txt subdirectories', required=True)
parser.add_argument('--dataset_out_dir', type=Path, help='The dataset output directory', required=True)
parser.add_argument('--force', '-f', action='store_true', help='force to overwrite existing files')
parser.add_argument('--debug_output_dir', type=Path, default=None, help='If specified we will write debug images to this directory')
parser.add_argument('--subset', default='test', help='train, val, test', choices=['train', 'val', 'test'])
args = parser.parse_args()

src_txt_dir = args.input_dir / 'txt'
src_img_dir = args.input_dir / 'orig'
dataset_dir = args.dataset_out_dir
json_dir = dataset_dir / 'Annos' / args.subset
json_dir.mkdir(parents=True, exist_ok=True)
dataset_img_dir = dataset_dir / 'Images' / args.subset
dataset_img_dir.mkdir(parents=True, exist_ok=True)
datalist_dir = dataset_dir / 'Datalist'
datalist_dir.mkdir(parents=True, exist_ok=True)

loader = DocBankLoader(txt_dir=src_txt_dir, img_dir=src_img_dir)
converter = DocBankConverter(loader)

examples = src_txt_dir.glob('**/*.txt')
examples = [os.path.basename(per) for per in examples]

datalist_json = {}

# class_labels = ['text', 'title', 'list', 'table', 'figure']
class_labels=(
	'abstract', 'author', 'caption', 'date', 'equation', 'figure', 'footer', 'list', 'paragraph', 'reference',
	'section', 'table', 'title')

def worker(example):
	example = loader.get_by_filename(example)

	# filter not processed file.
	save_name = json_dir / os.path.basename(example.filepath).replace('.jpg', '.json')
	if not os.path.exists(save_name) or args.force:
		print(save_name)

		formatted_json = {}
		formatted_json['height'] = example.pagesize[1]
		formatted_json['width'] = example.pagesize[0]

		content_ann = {}
		content_ann2 = {}

		## token level
		bboxes = example.denormalized_bboxes()
		filepath = example.filepath  # The image filepath
		pagesize = example.pagesize  # The image size
		words = example.words  # The tokens
		# bboxes = example.bboxes  # The normalized bboxes
		rgbs = example.rgbs  # The RGB values
		fontnames = example.fontnames  # The fontnames
		structures = example.structures  # The structure labels

		labels_list = [[class_labels.index(per)] for per in structures]
		attributes_list = [[font, rgb[0], rgb[1], rgb[2]] for font, rgb in zip(fontnames, rgbs)]

		content_ann['bboxes'] = bboxes
		content_ann['texts'] = words
		content_ann['labels'] = labels_list
		content_ann['attributes'] = attributes_list
		content_ann['cares'] = [1]*len(attributes_list)

		# layout level
		layout_examples = converter.get_by_filename(os.path.basename(filepath))
		layout_bboxes = layout_examples.print_bbox().split('\n')
		layout_bboxes = [per.split('\t') for per in layout_bboxes]
		layout_bboxes_list = []
		layout_labels_list = []

		for per_bbox in layout_bboxes:
			layout_bboxes_list.append([int(per_bbox[0]), int(per_bbox[1]), int(per_bbox[2]), int(per_bbox[3])])
			layout_labels_list.append([class_labels.index(per_bbox[4])])

		content_ann2['bboxes'] = layout_bboxes_list
		content_ann2['labels'] = layout_labels_list
		content_ann2['cares'] = [1]*len(layout_bboxes_list)

		formatted_json['content_ann'] = content_ann
		formatted_json['content_ann2'] = content_ann2

		# json output
		save_name = json_dir /  os.path.basename(filepath).replace('.jpg', '.json')
		if not os.path.exists(os.path.dirname(save_name)):
			os.makedirs(os.path.dirname(save_name))
		with open(save_name, 'w', encoding='utf8') as wf:
			json.dump(formatted_json, wf)

		# update datalist dict
		dataset_img_path = dataset_img_dir / os.path.basename(filepath)
		datalist_json[str(Path(*dataset_img_path.parts[-3:]))] = {
			'height': formatted_json['height'],
			'width': formatted_json['width'],
			'url': os.path.basename(save_name)
		}

		# Copy image to dataset directory
		shutil.copy(filepath, dataset_img_dir / os.path.basename(filepath))

		# visualize
		if args.debug_output_dir:
			debug_output_dir = args.debug_output_dir
			debug_output_dir.mkdir(parents=True, exist_ok=True)

			color_map = {
				'paragraph': (255, 0, 0),
				'section': (0, 255, 0),
				'list': (0, 0, 255),
				'abstract': (0, 255, 255),
				'author': (255, 0, 255),
				'equation': (255, 255, 0),
				'figure': (128, 0, 0),
				'table': (0, 128, 0),
				'title': (0, 0, 128),
			}
			img = cv2.imread(filepath)
			layout_img = copy.deepcopy(img)
			bboxes = content_ann['bboxes']
			labels = content_ann['labels']

			for idx, per_bbox in enumerate(bboxes):
				color = color_map[labels[idx][0]] if labels[idx][0] in color_map else (0, 0, 0)
				cv2.rectangle(img, (per_bbox[0], per_bbox[1]), (per_bbox[2], per_bbox[3]), color)

			layout_bboxes = content_ann2['bboxes']
			layout_labels = content_ann2['labels']
			for idx, per_bbox in enumerate(layout_bboxes):
				color = color_map[layout_labels[idx][0]] if layout_labels[idx][0] in color_map else (0, 0, 0)
				cv2.rectangle(layout_img, (int(per_bbox[0]), int(per_bbox[1])), (int(per_bbox[2]), int(per_bbox[3])),
				              color)

			cv2.imwrite(str(debug_output_dir / Path(filepath).name), np.concatenate((img, layout_img), 1))

# ## single process
# for example in tqdm(examples):
# 	worker(example)

## multiple processes
pool = multiprocessing.Pool(processes=4)
for example in tqdm(examples):
	#pool.apply_async(worker, (example,))
	worker(example)
pool.close()
pool.join()

# write datalist JSON
with open(datalist_dir / f"datalist_{args.subset}.json", 'w', encoding='utf8') as wf:
	json.dump(datalist_json, wf, indent=4, sort_keys=True)