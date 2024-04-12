#!/usr/bin/env bash
#set -eo pipefail
# ----------------------------------------------------------------
# @time: 2024/04/07 17:22
# @author: Wang Ke
# @contact: wangke09@58.com
# ----------------------------------------------------------------

PYTHONPATH=. python run_inference.py '240407_114745_600171245848pic_png' 'Unembellished face, formal wear, formal clothes, identification photo, ID photo, raw photo, masterpiece, chinese, solo, medium shot, high detail face, looking straight into the camera with shoulders parallel to the frame, photorealistic, best quality' 2>&1 | tee -a "logs__$(date +%y%m%d_%H%M%S)run_inference.log"
rsync -avz --password-file=<(echo "123") ./data/cache_req/240407_114745_600171245848pic_png/output_generated/* dkp@10.253.69.198::dkp
