#!/usr/bin/env bash
set -eo pipefail
# ----------------------------------------------------------------
# @time: 2024/04/08 14:39
# @author: Wang Ke
# @contact: wangke09@58.com
# ----------------------------------------------------------------

rsync -avzr --perms --chmod=a+rwx --include={*/,*.py,*.sh} --exclude="*" dk@10.253.69.198::dk/working/facechain/* .
rm -rf /usr/local/lib/python3.10/dist-packages/mmcv/*
#cp -r /usr/local/lib/python3.10/dist-packages/mmcv_bak_u1/* /usr/local/lib/python3.10/dist-packages/mmcv/
cp -r /usr/local/lib/python3.10/dist-packages/mmcv_bak_raw/* /usr/local/lib/python3.10/dist-packages/mmcv/

ps -ef | grep "seldon-core" | grep -v "grep" | awk '{print $2}' | xargs kill -9
nohup seldon-core-microservice predictor GRPC --service-type MODEL --persistence 0 --workers 10 2>&1 >logs__$(date +%y%m%d_%H%M%S)predictor.log & ## 启动程序, 在processor.py 同一级目录执行
tail -f $(ls -lrt | grep -F '.log' | awk '{print $NF}' | tail -n 1)
