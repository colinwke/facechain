#!/usr/bin/env bash
set -eo pipefail
# ----------------------------------------------------------------
# @time: 2024/04/03 14:50
# @author: Wang Ke
# @contact: wangke09@58.com
# ----------------------------------------------------------------
date1=$(date +%s)

rm -rf *.log
rm -rf ./data

version=facegen_v02
project_dir="/code/dkc/project/${version}"

rm -rf ${project_dir}
mkdir -p ${project_dir}
cp -r . ${project_dir}

du -sh ${project_dir}
echo "[project_dir] ${project_dir}"
echo "[elapse] $(($(date +%s) - date1)) seconds!"
