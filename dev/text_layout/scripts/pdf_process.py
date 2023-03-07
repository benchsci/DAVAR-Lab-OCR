# Adapted from https://github.com/doc-analysis/DocBank/blob/master/scripts/pdf_process.py

import multiprocessing
import argparse
import pdfplumber
import os
from tqdm import tqdm
from pdfminer.layout import LTChar, LTLine
import re
from collections import Counter
import pdf2image
import numpy as np
from PIL import Image
from pathlib import Path
import json


def within_bbox(bbox_bound, bbox_in):
    assert bbox_bound[0] <= bbox_bound[2]
    assert bbox_bound[1] <= bbox_bound[3]
    assert bbox_in[0] <= bbox_in[2]
    assert bbox_in[1] <= bbox_in[3]

    x_left = max(bbox_bound[0], bbox_in[0])
    y_top = max(bbox_bound[1], bbox_in[1])
    x_right = min(bbox_bound[2], bbox_in[2])
    y_bottom = min(bbox_bound[3], bbox_in[3])

    if x_right < x_left or y_bottom < y_top:
        return False

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox_in_area = (bbox_in[2] - bbox_in[0]) * (bbox_in[3] - bbox_in[1])

    if bbox_in_area == 0:
        return False

    iou = intersection_area / float(bbox_in_area)

    return iou > 0.95


def worker(
        pdf_file: str,
        page_no: int,
        output_dir: str,
        unified_output_dir: bool = False,
        max_image_dim: int = None,
        use_page_subdirs: bool = False,
        gt_annotations: dict = None
):


    print(f"Processing {pdf_file}")
    try:
        pdf_images = pdf2image.convert_from_path(pdf_file)
    except Exception as e:
        print(e)
        return
    output_dir = Path(output_dir)

    page_tokens = []
    try:
        pdf = pdfplumber.open(pdf_file)
    except Exception as e:
        print(e)
        return

    if len(pdf.pages) > 1:
        ValueError(f"Warning: {pdf_file} has more than 1 page")

    for page_id in tqdm(range(len(pdf.pages))):
        tokens = []
        print(f"Processing {pdf_file}:p{page_id}")

        this_page = pdf.pages[page_id]
        anno_img = np.ones([int(this_page.width), int(this_page.height)] + [3], dtype=np.uint8) * 255

        words = this_page.extract_words(x_tolerance=1.5)

        lines = []
        for obj in this_page.layout._objs:
            if not isinstance(obj, LTLine):
                continue
            lines.append(obj)

        for word in words:
            word_bbox = (float(word['x0']), float(word['top']), float(word['x1']), float(word['bottom']))
            objs = []
            for obj in this_page.layout._objs:
                if not isinstance(obj, LTChar):
                    continue
                obj_bbox = (obj.bbox[0], float(this_page.height) - obj.bbox[3],
                            obj.bbox[2], float(this_page.height) - obj.bbox[1])
                if within_bbox(word_bbox, obj_bbox):
                    objs.append(obj)
            fontname = []
            for obj in objs:
                fontname.append(obj.fontname)
            if len(fontname) != 0:
                c = Counter(fontname)
                fontname, _ = c.most_common(1)[0]
            else:
                fontname = 'default'

            # format word_bbox
            width = int(this_page.width)
            height = int(this_page.height)
            f_x0 = min(1000, max(0, int(word_bbox[0] / width * 1000)))
            f_y0 = min(1000, max(0, int(word_bbox[1] / height * 1000)))
            f_x1 = min(1000, max(0, int(word_bbox[2] / width * 1000)))
            f_y1 = min(1000, max(0, int(word_bbox[3] / height * 1000)))
            word_bbox = tuple([f_x0, f_y0, f_x1, f_y1])

            # plot annotation
            x0, y0, x1, y1 = word_bbox
            x0, y0, x1, y1 = int(x0 * width / 1000), int(y0 * height / 1000), int(x1 * width / 1000), int(
                y1 * height / 1000)
            anno_color = [0, 0, 0]
            for x in range(x0, x1):
                for y in range(y0, y1):
                    anno_img[x, y] = anno_color

            word_bbox = tuple([str(t) for t in word_bbox])
            word_text = re.sub(r"\s+", "", word['text'])
            tokens.append((word_text,) + word_bbox + (fontname,))

        try:
            figures = this_page.images
        except AttributeError:
            print(f"Warning: no figures found on {pdf_file}:p{page_id}")
            figures = []

        for figure in figures:
            figure_bbox = (float(figure['x0']), float(figure['top']), float(figure['x1']), float(figure['bottom']))

            # format word_bbox
            width = int(this_page.width)
            height = int(this_page.height)
            f_x0 = min(1000, max(0, int(figure_bbox[0] / width * 1000)))
            f_y0 = min(1000, max(0, int(figure_bbox[1] / height * 1000)))
            f_x1 = min(1000, max(0, int(figure_bbox[2] / width * 1000)))
            f_y1 = min(1000, max(0, int(figure_bbox[3] / height * 1000)))
            figure_bbox = tuple([f_x0, f_y0, f_x1, f_y1])

            # plot annotation
            x0, y0, x1, y1 = figure_bbox
            x0, y0, x1, y1 = int(x0 * width / 1000), int(y0 * height / 1000), int(x1 * width / 1000), int(
                y1 * height / 1000)
            anno_color = [0, 0, 0]
            for x in range(x0, x1):
                for y in range(y0, y1):
                    anno_img[x, y] = anno_color

            figure_bbox = tuple([str(t) for t in figure_bbox])
            word_text = '##LTFigure##'
            fontname = 'default'
            tokens.append((word_text,) + figure_bbox + (fontname,))

        # for line in this_page.lines:
        #     line_bbox = (float(line['x0']), float(line['top']), float(line['x1']), float(line['bottom']))
        #     # format word_bbox
        #     width = int(this_page.width)
        #     height = int(this_page.height)
        #     f_x0 = min(1000, max(0, int(line_bbox[0] / width * 1000)))
        #     f_y0 = min(1000, max(0, int(line_bbox[1] / height * 1000)))
        #     f_x1 = min(1000, max(0, int(line_bbox[2] / width * 1000)))
        #     f_y1 = min(1000, max(0, int(line_bbox[3] / height * 1000)))
        #     line_bbox = tuple([f_x0, f_y0, f_x1, f_y1])

        #     # plot annotation
        #     x0, y0, x1, y1 = line_bbox
        #     x0, y0, x1, y1 = int(x0 * width / 1000), int(y0 * height / 1000), int(x1 * width / 1000), int(
        #         y1 * height / 1000)
        #     anno_color = [0, 0, 0]
        #     for x in range(x0, x1 + 1):
        #         for y in range(y0, y1 + 1):
        #             anno_img[x, y] = anno_color

        #     line_bbox = tuple([str(t) for t in line_bbox])
        #     word_text = '##LTLine##'
        #     fontname = 'default'
        #     tokens.append((word_text,) + line_bbox + (fontname, ))

        anno_img = np.swapaxes(anno_img, 0, 1)
        anno_img = Image.fromarray(anno_img, mode='RGB')
        page_tokens.append((page_id, tokens, anno_img))

        if unified_output_dir:
            txt_output_dir = output_dir
            orig_output_dir = output_dir
            anno_output_dir = output_dir
        else:
            txt_output_dir = output_dir / 'txt'
            orig_output_dir = output_dir / 'orig'
            anno_output_dir = output_dir / 'anno'

        imwidth, imheight = pdf_images[page_id].size
        max_dim = max(imwidth, imheight)
        if max_image_dim is not None and max_dim > max_image_dim:
            sf = max_image_dim / max_dim
            new_width = int(imwidth * sf)
            new_height = int(imheight * sf)
            print(f"Resizing {pdf_file} from {imwidth}x{imheight} to {new_width}x{new_height}")
            pdf_images[page_id] = pdf_images[page_id].resize((new_width, new_height), Image.ANTIALIAS)

        doc_fname = pdf_file.replace('.pdf', '')

        if use_page_subdirs:
            txt_output_dir = txt_output_dir / doc_fname
            orig_output_dir = orig_output_dir / doc_fname
            anno_output_dir = anno_output_dir / doc_fname

        txt_output_dir.mkdir(parents=True, exist_ok=True)
        orig_output_dir.mkdir(parents=True, exist_ok=True)
        anno_output_dir.mkdir(parents=True, exist_ok=True)

        pdf_images[page_id].save(orig_output_dir / Path(doc_fname + '_{}_ori.jpg'.format(str(page_id))).name)
        anno_img.save(anno_output_dir / Path(doc_fname + '_{}_ann.jpg'.format(str(page_id))).name)
        with open(txt_output_dir /  Path(doc_fname + '_{}.txt'.format(str(page_id))).name,
                    'w',
                    encoding='utf8') as fp:
            for token in tokens:
                fp.write('\t'.join(token) + '\n')
        # save GT
        with open(gt_output_dir / Path(doc_fname + '_{}_gt.json'.format(str(page_id))).name,
                    'w',
                    encoding='utf8') as fp:
            json.dump(gt_annotations, fp, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    mutex_input_group = parser.add_mutually_exclusive_group()
    mutex_input_group.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the pdf files.",
    )
    input_group2 = mutex_input_group.add_argument_group()
    input_group2.add_argument(
        "--pdf_list_file",
        help="A file containing a list of pdf files to process.",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the output data will be written.",
    )
    parser.add_argument(
        "--max_image_dim",
        default=None,
        type=int,
        help="The maximum image dimension to use for the image. If the image is larger than this, it will be scaled down.",
    )
    parser.add_argument(
        "--use_page_subdirs",
        help="Whether to use subdirectories for separate pages from the same document",
        action="store_true"
    )
    parser.add_argument(
        "--unified_output_dir",
        action='store_true',
        help="Whether to use the same output directory for all output file types",
    )
    parser.add_argument(
        "--limit",
        help="Limit the number of pdf files to process",
        type=int,
        default=None
    )
    args = parser.parse_args()


    if args.data_dir is not None:
        pdf_files = list(os.listdir(args.data_dir))
        pdf_files = [(t, None) for t in pdf_files if t.endswith('.pdf')]
    elif args.pdf_list_file is not None:
        with open(args.pdf_list_file, 'r') as fp:
            pdf_files = [line.strip().split(',') for line in fp]
            # Skip header
            pdf_files = pdf_files[1:]

    if args.limit is not None:
        pdf_files = pdf_files[:args.limit]

    pool = multiprocessing.Pool(processes=4)
    for _, (pdf_file, page_no) in enumerate(tqdm(pdf_files)):
        # pool.apply_async(worker, (pdf_file, args.data_dir, args.output_dir, args.unified_output_dir, args.max_image_dim, args.use_page_subdirs))
        page_no = int(page_no) if page_no is not None else None
        key = Path(pdf_file).stem
        worker(pdf_file, page_no, args.output_dir, args.unified_output_dir, args.max_image_dim, args.use_page_subdirs)

    pool.close()
    pool.join()
