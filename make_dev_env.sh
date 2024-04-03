#!/usr/bin/env bash
set -eo pipefail
# ----------------------------------------------------------------
# @time: 2024/04/02 17:07
# @author: Wang Ke
# @contact: wangke09@58.com
# ----------------------------------------------------------------

sh make_rsync_daemon_works.sh

mkdir -p /workspace/model/facegen_tc212 && cd /workspace/model/facegen_tc212
rsync -avzrP --perms --chmod=a+rwx dk@10.253.69.198::dk/working/facechain/* .  # local to online




