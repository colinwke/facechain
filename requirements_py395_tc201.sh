#!/usr/bin/env bash
# shellcheck disable=SC2046
echo "1015"
#shellcheck disable=SC2046
#set -eo pipefail
# ----------------------------------------------------------------
# @time: 2024/03/14 15:05
# @author: Wang Ke
# @contact: wangke09@58.com
# ----------------------------------------------------------------
USAGE=$( # this is a block comment
    cat <<'EOF'
rsync -avz dk@10.253.69.198::dk/working/facechain/requirements_py395_tc201.sh .
hrmr /home/hdp_lbg_ectech/resultdata/wangke/script/a/requirements_py395_tc201.sh
hadoop fs -put requirements_py395_tc201.sh /home/hdp_lbg_ectech/resultdata/wangke/script/a/
hld /home/hdp_lbg_ectech/resultdata/wangke/script/a/

mkdir -p /tmp && cd /tmp && hadoop fs -get /home/hdp_lbg_ectech/resultdata/wangke/script/a/requirements_py395_tc201.sh && /bin/bash requirements_py395_tc201.sh

# CUDA 11.8  ## https://pytorch.org/get-started/previous-versions/#v200
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

yes | pip3 uninstall $(pip3 list | grep 'torch' | awk '{print $1}')
yes | pip3 uninstall $(pip3 list | grep 'cuda' | awk '{print $1}')

pip list | grep torch
torch                                    2.1.0
torchaudio                               2.1.0
torchdata                                0.7.0
torchtext                                0.16.0
torchvision                              0.16.0
pip list | grep cuda
nvidia-cuda-cupti-cu12                   12.1.105
nvidia-cuda-nvrtc-cu12                   12.1.105
nvidia-cuda-runtime-cu12                 12.1.105


pip install --force-reinstall -v "MySQL_python==1.2.2"

EOF
)

# ----------------------------------------------------------------
# ----------------------------------------------------------------

# https://mirrors.tuna.tsinghua.edu.cn/help/pypi/

echo "RUNNING scipt $0"
python --version

PROJECT_DIR="/code/dkc/project/facegen_tc201"
if [[ -d "${PROJECT_DIR}" ]]; then
    #镜像环境构建不支持指定镜像源, 否则会构建失败
    #python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
    pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
fi

yes | pip3 uninstall $(pip3 list | grep 'torch' | awk '{print $1}')
yes | pip3 uninstall $(pip3 list | grep 'cuda' | awk '{print $1}')

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

yes | pip3 uninstall chromadb continuedev pandas pywavelets vllm xformers

# pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118  ## official very slow
pip3 install torch==2.0.1+cu118 torchvision torchaudio -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html ## mirror faster than official
pip3 install numpy==1.22.0
# pip3 install --force-reinstall numpy==1.22.0
pip3 install chromadb continuedev pandas pywavelets # vllm xformers  ## `vllm xformers` not need, and xformer and vllm need torch==2.2.1

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

#pip3 install mmcv-full  # install with setup.py, very slow
pip3 install -U openmim
mim install mmcv-full

pip3 install accelerate
pip3 install transformers
pip3 install Pillow
pip3 install opencv-python
pip3 install torchvision
pip3 install mmengine
pip3 install timm
pip3 install mediapipe
pip3 install python-slugify
pip3 install edge-tts

pip3 install numpy==1.22.0 # https://endoflife.date/numpy
pip3 install pandas==1.4.4 # gradio dependence
# pandas version with date: https://pandas.pydata.org/docs/whatsnew/index.html

yes | pip3 uninstall $(pip3 list | grep 'cuda.*12' | awk '{print $1}')
pip3 list | grep -E 'cuda|diffusers|gradio|mmcv|mmdet|modelscope|numpy|onnxruntime|pandas|protobuf|torch|MarkupSafe|Jinja2'
