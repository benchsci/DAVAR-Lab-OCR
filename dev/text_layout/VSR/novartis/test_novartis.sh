#!/bin/bash
export LANG=zh_CN.UTF-8
export LANGUAGE=zh_CN:zh:en_US:en
export PATH=/usr/local/miniconda3/bin/:$PATH

export CUDA_VISIBLE_DEVICES=0

DAVAROCR_PATH=/home/jupyter/kweston_persistent_workspace/DAVAR-Lab-OCR
DATASET_DIR=${DAVAROCR_PATH}/dev/text_layout/datalist/NovartisCustom

#rm -rf pdfout
#rm -rf $DATASET_DIR
#python ~/kweston_persistent_workspace/DocBank/scripts/pdf_process.py --data_dir pdf_in --output_dir pdf_out
#python ../../scripts/format_dataset.py --input_dir pdf_out --dataset_out_dir $DATASET_DIR -f --debug_output_dir $DATASET_DIR/debug_out
#ln -s ~/kweston_persistent_workspace/DAVAR-Lab-OCR/demo/text_layout/datalist/PubLayNet/coco_val.json $DATASET_DIR/coco_val.json
python $DAVAROCR_PATH/tools/test.py  ./config/novartis_docbank_style_config.py ./weights/DocBank/docbank_x101-eb65a9b1.pth --eval bbox --show-dir showdir