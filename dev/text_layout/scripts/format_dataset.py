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
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass


dirname = os.path.join(os.path.dirname(__file__), '../../../demo/text_layout/datalist/DocBank/Scripts/DocBankLoader-master')
sys.path.append(dirname)

from docbank_loader import DocBankLoader, DocBankConverter

# txt_dir = 'demo/txt'
# img_dir = 'demo/img'



def xywh2xyxy(bbox: List[float]) -> List[float]:
	x, y, w, h = bbox
	return x, y, x + w, y + h

def normalize_bbox(bbox: List[float], image_width: float, image_hight: float) -> List[float]:
	x1, y1, x2, y2 = bbox
	return x1 / image_width, y1 / image_hight, x2 / image_width, y2 / image_hight

def rescale_bbox(bbox: List[float], image_width: float, image_hight: float) -> List[float]:
	x1, y1, x2, y2 = bbox
	return x1 * image_width, y1 * image_hight, x2 * image_width, y2 * image_hight

def get_layout_pseudo_gt(filepath: str) -> Tuple[List, List]:

	layout_examples = converter.get_by_filename(os.path.basename(filepath))
	layout_bboxes = layout_examples.print_bbox().split('\n')
	layout_bboxes = [per.split('\t') for per in layout_bboxes]
	layout_bboxes_list = []
	layout_labels_list = []
	for per_bbox in layout_bboxes:
		layout_bboxes_list.append([int(per_bbox[0]), int(per_bbox[1]), int(per_bbox[2]), int(per_bbox[3])])
		layout_labels_list.append([class_labels.index(per_bbox[4])])

	return layout_bboxes_list, layout_labels_list

def get_layout_gt(filepath: Path, annotations: dict, imwidth: int, imheight: int) -> Tuple[List, List]:

	key = Path(filepath).stem.replace('_0_ori', '')
	anns = annotations[key]

	layout_bboxes_list = []
	layout_labels_list = []
	orig_imwidth = anns['width']
	orig_imheight = anns['height']

	for ann in anns['objects']:
		bbox = map(float, ann['bbox'])
		bbox = xywh2xyxy(bbox)
		bbox = normalize_bbox(bbox, orig_imwidth, orig_imheight)
		bbox = rescale_bbox(bbox, imwidth, imheight)
		layout_bboxes_list.append(bbox)
		layout_labels_list.append(int(ann['category_id']))
	return layout_bboxes_list, layout_labels_list

# class_labels = ['text', 'title', 'list', 'table', 'figure']
# class_labels=(
# 	'abstract', 'author', 'caption', 'date', 'equation', 'figure', 'footer', 'list', 'paragraph', 'reference',
# 	'section', 'table', 'title')

def worker(example, class_labels: List[str], dataset_annos_dir: Path, dataset_img_dir: Path, gt_annotations: Optional[Dict], colormap: Optional[Dict[str, Tuple[int, int, int]]]):
	example = loader.get_by_filename(example)

	# filter not processed file.
	save_name = dataset_annos_dir / os.path.basename(example.filepath).replace('.jpg', '.json')
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

		labels_list = [[class_labels.index(per)] if per is not None else None for per in structures]
		attributes_list = [[font, rgb[0], rgb[1], rgb[2]] for font, rgb in zip(fontnames, rgbs)]

		content_ann['bboxes'] = bboxes
		content_ann['texts'] = words
		content_ann['labels'] = labels_list
		content_ann['attributes'] = attributes_list
		content_ann['cares'] = [1]*len(attributes_list)

		if gt_annotations is None:
			layout_bboxes_list, layout_labels_list = get_layout_pseudo_gt(filepath)
		else:
			img = cv2.imread(filepath)
			imheight, imwidth = img.shape[:2]
			layout_bboxes_list, layout_labels_list = get_layout_gt(filepath, gt_annotations, imwidth, imheight)

		content_ann2['bboxes'] = layout_bboxes_list
		content_ann2['labels'] = layout_labels_list
		content_ann2['cares'] = [1]*len(layout_bboxes_list)

		formatted_json['content_ann'] = content_ann
		formatted_json['content_ann2'] = content_ann2

		# json output
		save_name = dataset_annos_dir /  os.path.basename(filepath).replace('.jpg', '.json')
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

			layout_img = copy.deepcopy(img)
			bboxes = content_ann['bboxes']
			labels = content_ann['labels']

			for idx, per_bbox in enumerate(bboxes):
				color = colormap[labels[idx]] if labels[idx] in colormap else (0, 0, 0)
				cv2.rectangle(img, (per_bbox[0], per_bbox[1]), (per_bbox[2], per_bbox[3]), color)

			layout_bboxes = content_ann2['bboxes']
			layout_labels = content_ann2['labels']
			for idx, bbox in enumerate(layout_bboxes):
				color = colormap[layout_labels[idx]] if layout_labels[idx] in colormap else (0, 0, 0)
				cv2.rectangle(layout_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
							color)

			cv2.imwrite(str(debug_output_dir / Path(filepath).name), np.concatenate((img, layout_img), 1))

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=Path, help='The output directory of pdf_converter.py containing both orig and txt subdirectories', required=True)
	parser.add_argument('--gt_annotations', type=Path, help='A JSON file containing ground truth annotations', default=None)
	parser.add_argument('--dataset_out_dir', type=Path, help='The dataset output directory', required=True)
	parser.add_argument('--subset', default='test', help='train, val, test', choices=['train', 'val', 'test'])
	parser.add_argument('--force', '-f', action='store_true', help='force to overwrite existing files')
	parser.add_argument('--debug_output_dir', type=Path, default=None, help='If specified we will write debug images to this directory')
	args = parser.parse_args()

	src_txt_dir = args.input_dir / 'txt'
	src_img_dir = args.input_dir / 'orig'
	dataset_dir = args.dataset_out_dir
	dataset_annos_dir = dataset_dir / 'Annos' / args.subset
	dataset_annos_dir.mkdir(parents=True, exist_ok=True)
	dataset_img_dir = dataset_dir / 'Images' / args.subset
	dataset_img_dir.mkdir(parents=True, exist_ok=True)
	datalist_dir = dataset_dir / 'Datalist'
	datalist_dir.mkdir(parents=True, exist_ok=True)

	loader = DocBankLoader(txt_dir=src_txt_dir, img_dir=src_img_dir)
	converter = DocBankConverter(loader)

	examples = src_txt_dir.glob('**/*.txt')
	examples = [os.path.basename(per) for per in examples]

	datalist_json = {}

	with open("class_names.json", "r") as f:
		class_labels = json.load(f)

	class_label_index = {per: idx for idx, per in enumerate(class_labels)}

	# Open the ground truth annotations file
	if args.gt_annotations:
		with open(args.gt_annotations, 'r', encoding='utf8') as f:
			gt_annotations = json.load(f)

	# ## single process
	# for example in tqdm(examples):
	# 	worker(example)
	# colormap = {
	# 	'paragraph': (255, 0, 0),
	# 	'section': (0, 255, 0),
	# 	'list': (0, 0, 255),
	# 	'abstract': (0, 255, 255),
	# 	'author': (255, 0, 255),
	# 	'equation': (255, 255, 0),
	# 	'figure': (128, 0, 0),
	# 	'table': (0, 128, 0),
	# 	'title': (0, 0, 128),
	# }
	colormap = defaultdict(lambda: (0, 0, 0))
	for label in range(len(class_labels)):
		colormap[label] = np.random.randint(0, 256, 3).tolist()


	## multiple processes
	pool = multiprocessing.Pool(processes=4)
	for example in tqdm(examples):
		#pool.apply_async(worker, (example,))
		worker(example, class_labels, dataset_annos_dir, dataset_img_dir, gt_annotations, colormap)
	pool.close()
	pool.join()

	# write datalist JSON
	with open(datalist_dir / f"datalist_{args.subset}.json", 'w', encoding='utf8') as wf:
		json.dump(datalist_json, wf, indent=4, sort_keys=True)