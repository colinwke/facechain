#!/usr/bin/env bash
#set -eo pipefail
# ----------------------------------------------------------------
# @time: 2024/03/14 15:05
# @author: Wang Ke
# @contact: wangke09@58.com
# ----------------------------------------------------------------
USAGE=$( # this is a block comment
    cat <<'EOF'
mkdir -p /tmp && cd /tmp && hadoop fs -get /home/hdp_lbg_ectech/resultdata/wangke/script/a/install_facechain_env.sh && /bin/bash install_facechain_env.sh

rsync -avz dk@10.253.69.198::dk/working/facechain/install_facechain_env.sh .
hrmr /home/hdp_lbg_ectech/resultdata/wangke/script/a/install_facechain_env.sh
hadoop fs -put install_facechain_env.sh /home/hdp_lbg_ectech/resultdata/wangke/script/a/
hld /home/hdp_lbg_ectech/resultdata/wangke/script/a/

EOF
)

#
#python -m pip install --upgrade pip
#python -m pip install -U pip

pip install --upgrade pip
#yes | pip3 uninstall torch
#yes | pip3 uninstall torchvision
#yes | pip3 uninstall torchaudio
yes | pip3 uninstall accelerate
yes | pip3 uninstall transformers
yes | pip3 uninstall onnxruntime
yes | pip3 uninstall diffusers
yes | pip3 uninstall invisible-watermark
yes | pip3 uninstall modelscope
yes | pip3 uninstall Pillow
yes | pip3 uninstall opencv-python
yes | pip3 uninstall torchvision
yes | pip3 uninstall mmdet
yes | pip3 uninstall mmengine
yes | pip3 uninstall numpy
yes | pip3 uninstall protobuf
yes | pip3 uninstall timm
yes | pip3 uninstall scikit-image
yes | pip3 uninstall gradio
yes | pip3 uninstall controlnet_aux
yes | pip3 uninstall mediapipe
yes | pip3 uninstall python-slugify
yes | pip3 uninstall edge-tts

pip3 uninstall chromadb continuedev pandas pywavelets vllm xformers
pip3 install numpy==1.22.0
# pip3 install --force-reinstall numpy==1.22.0
pip3 install chromadb continuedev pandas pywavelets vllm xformers

# mmcv-full (need mim install)
pip3 install protobuf==3.20.1
pip3 install scikit-image==0.19.3
#pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

pip3 install gradio==3.50.2
pip3 install controlnet_aux==0.0.6
pip3 install onnxruntime==1.15.1
pip3 install modelscope==1.10.0

pip3 install onnxruntime==1.15.1
pip3 install diffusers==0.23.0
pip3 install invisible-watermark==0.2.0
pip3 install modelscope==1.10.0
pip3 install mmdet==2.26.0
pip3 install controlnet_aux==0.0.6

pip3 install python-slugify
pip3 install edge-tts

pip3 install mmcv-full
# pip3 install -U openmim
# mim install mmcv-full
pip3 install accelerate
pip3 install transformers
pip3 install Pillow
pip3 install opencv-python
pip3 install torchvision
pip3 install mmengine
pip3 install timm
# pip3 install gradio
pip3 install mediapipe
pip3 install python-slugify
pip3 install edge-tts

pip3 install numpy==1.22.0 # https://endoflife.date/numpy
pip3 install pandas==1.4.4 # gradio dependence
# pandas version with date: https://pandas.pydata.org/docs/whatsnew/index.html

pip list | grep -E 'diffusers|gradio|mmcv|mmdet|modelscope|numpy|onnxruntime|pandas|protobuf|torch'
