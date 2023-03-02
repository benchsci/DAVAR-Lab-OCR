#!/bin/bash
export LANG=zh_CN.UTF-8
export LANGUAGE=zh_CN:zh:en_US:en
export PATH=/usr/local/miniconda3/bin/:$PATH

export CUDA_VISIBLE_DEVICES=0

DAVAROCR_PATH=/home/jupyter/kweston_persistent_workspace/DAVAR-Lab-OCR
DATASET_DIR=${DAVAROCR_PATH}/dev/text_layout/datalist/NovartisCustom
#INPUT_DIR=pdf_in
INPUT_DIR=/home/jupyter/kweston_persistent_workspace/novartis_validation_data/361_ELNs_to_focus_on
OUTPUT_DIR=showdir2

rm -rf pdf_out
rm -rf $OUTPUT_DIR
rm -rf $DATASET_DIR
echo "Running pdf_process.py"
python ../../scripts/pdf_process.py --data_dir $INPUT_DIR --output_dir pdf_out --limit 100
echo "Running format_dataset.py"
python ../../scripts/format_dataset.py --input_dir pdf_out --dataset_out_dir $DATASET_DIR -f --debug_output_dir $DATASET_DIR/debug_out
ln -s $DAVAROCR_PATH/demo/text_layout/datalist/DocBank/Datalist/classes_config.json $DATASET_DIR/Datalist/classes_config.json
echo "Running test.py"
python $DAVAROCR_PATH/tools/test.py ./config/novartis_docbank_style_config.py $DAVAROCR_PATH/weights/DocBank/docbank_x101-eb65a9b1.pth --format-only --show-dir $OUTPUT_DIR