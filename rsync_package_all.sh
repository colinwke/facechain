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

date1=$(date +%s)

PROJECT_DIR="/code/dkc/project/facegen_tc212"
WORKSPACE_DIR="/workspace/model/facegen_tc212"
model_dir="${PROJECT_DIR}"

if [[ ! -d "${model_dir}" ]]; then
    echo "$model_dir does not exist. copy all project!"
    mkdir -p "${model_dir}"
    cp -r ${WORKSPACE_DIR}/* ${PROJECT_DIR}/
else
    model_size=$( (du -s "${model_dir}" || echo "0") | awk '{print $1}')
    check_size=1000000
    if [[ ${model_size} -lt ${check_size} ]]; then
        echo "[ERROR] model_dir( ${model_dir} ) exist and size over ${check_size}! delete first!" && exit 15
    else
        echo "$model_dir does exist. only rsync code!"
        rsync -avzr --perms --chmod=a+rwx --include={*/,*.py,*.sh} --exclude="*" ${WORKSPACE_DIR}/* ${PROJECT_DIR}/  # --delete not work
    fi
fi

echo -e "\n\n\n---"

project_size=$( (du -s "${PROJECT_DIR}" || echo "0") | awk '{print $1}')

cat <<-EOF
[size] $((project_size / (1000 * 1000)))G || ${project_size}
[path] ${PROJECT_DIR}
[elapse] $(($(date +%s) - date1)) seconds!

EOF
