#!/usr/bin/env bash
set -eo pipefail
# ----------------------------------------------------------------
# @time: 2024/03/19 17:13
# @author: Wang Ke
# @contact: wangke09@58.com
# ----------------------------------------------------------------
USAGE=$( # this is a block comment
    cat <<'EOF'
elapse of script with linux: https://stackoverflow.com/a/20249473/6494418

EOF
)



date1=$(date +"%s")


#rsync -avzrP --delete --perms --chmod=a+rwx /workspace/facegen/* /code/dkc/project/facegen/

#rsync -avz /workspace/facegen/* /code/dkc/project/facegen/
#rsync -avzr /workspace/facegen /code/dkc/project/
# rsync very slow

#rm -rf /code/dkc/project/facegen/*
#cp -r /workspace/facegen/* /code/dkc/project/facegen/





du -sh /code/dkc/project/facegen/

date2=$(date +"%s")
DIFF=$((date2 - date1))
echo "Duration: $((DIFF / 3600)) hours $(((DIFF % 3600) / 60)) minutes $((DIFF % 60)) seconds"
