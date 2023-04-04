#!/bin/bash
export LANG=zh_CN.UTF-8
export LANGUAGE=zh_CN:zh:en_US:en
export PATH=/usr/local/miniconda3/bin/:$PATH

export CUDA_VISIBLE_DEVICES=0

DAVAROCR_PATH=/home/jupyter/kweston_persistent_workspace/DAVAR-Lab-OCR
DATASET_DIR=${DAVAROCR_PATH}/dev/text_layout/datalist/DocLayNetSmall
INPUT_DIR=pdf_in
INPUT_DIR=/home/jupyter/kweston_persistent_workspace/novartis_validation_data/361_ELNs_to_focus_on
OUTPUT_DIR=showdir3
HUGGINGFACE_OUTDIR=tmp

rm -rf pdf_out
rm -rf $OUTPUT_DIR
rm -rf $DATASET_DIR
# echo "Running convert_huggingface_dataset.py"
# ./convert_huggingface_dataset.py $HUGGINGFACE_OUTDIR
echo "Running pdf_process.py"
python ../../scripts/pdf_process.py --pdf_list_file $HUGGINGFACE_OUTDIR/pdf_page_list.csv --output_dir pdf_out --limit 100
 echo "Running format_dataset.py"
python ../../scripts/format_dataset.py --input_txt_dir pdf_out/txt --input_img_dir $HUGGINGFACE_OUTDIR/debug --dataset_out_dir $DATASET_DIR -f --gt_annotation_file $HUGGINGFACE_OUTDIR/annotations.json --debug_output_dir $DATASET_DIR/debug_out
# ln -s $DAVAROCR_PATH/demo/text_layout/datalist/DocBank/Datalist/classes_config.json $DATASET_DIR/Datalist/classes_config.json
# echo "Running test.py"
# python $DAVAROCR_PATH/tools/test.py ./config/novartis_docbank_style_config.py $DAVAROCR_PATH/weights/DocBank/docbank_x101-eb65a9b1.pth --format-only --show-dir $OUTPUT_DIR