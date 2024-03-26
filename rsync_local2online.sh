#!/usr/bin/env bash
# shellcheck disable=SC2086
set -eo pipefail
# ----------------------------------------------------------------
# @time: 2024/03/20 20:57
# @author: Wang Ke
# @contact: wangke09@58.com
# ----------------------------------------------------------------

config_flag="code"
if [[ $# -ge 1 ]]; then
    config_flag=$1
fi

date1=$(date +"%s")
ip=10.253.69.198
from_path="dk/working/facechain/*"
to_path="."

echo "[config_flag] ${config_flag}"
if [[ "${config_flag}" == "full" ]]; then
    rsync -avzr dk@${ip}::${from_path} ${to_path} --exclude={.*/,_*/,out/,*__pycache__*/,.*,*.pyc,*.zip,*.log,log/,*log.txt,metastore_db,generated*,output*,processed*,input*} # offline to online
else
    # only code
    rsync -avzr --perms --chmod=a+rwx --include={*/,*.py,*.sh} --exclude="*" dk@${ip}::${from_path} ${to_path} # local to online
fi

echo "DONE"
date2=$(date +"%s")
DIFF=$((date2 - date1))
echo "Duration: $((DIFF / 3600)) hours $(((DIFF % 3600) / 60)) minutes $((DIFF % 60)) seconds"
