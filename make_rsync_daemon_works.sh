#!/usr/bin/env bash
set -eo pipefail
# ----------------------------------------------------------------
# @time: 2024/03/31 11:08
# @author: Wang Ke
# @contact: wangke09@58.com
# ----------------------------------------------------------------

# install rsync
if ! command -v rsync &>/dev/null; then
    apt-get update && apt-get upgrade -y && apt-get install -y rsync git ## install rsync
fi
if ! command -v ifconfig &>/dev/null; then
    apt-get update && apt-get upgrade -y && apt-get install net-tools ## # ifconfig
fi

mkdir -p /etc/rsyncd

# setting `rsyncd.conf`
cat <<'EOF' >"/etc/rsyncd/rsyncd.conf"
# pid file = /etc/rsyncd/rsyncd.pid
# log file = /etc/rsyncd/rsyncd.log
# lock file = /etc/rsyncd/rsyncd.lock
# conf file = /etc/rsyncd/rsyncd.conf
secrets file = /etc/rsyncd/rsyncd.secrets

uid = 0
gid = 0

use chroot = false
strict modes = no
max connections = 5

transfer logging = yes
read only = no
list = no

hosts allow = *
auth users = *

# --------------------------------
# Modules Config
# --------------------------------
[r]
path = /

#[facegen_tc212]
## path = /workspace/model/facegen_tc212
#path = /code/dkc/project/facegen_tc212


EOF

# setting `rsyncd.secrets`
cat <<'EOF' >"/etc/rsyncd/rsyncd.secrets"
dk:dk
EOF

/usr/bin/rsync --daemon --config=/etc/rsyncd/rsyncd.conf
ip=$(ifconfig | sed -En 's/127.0.0.1//;s/.*inet (addr:)?(([0-9]*\.){3}[0-9]*).*/\2/p' | tail -n 1 | tr -d '[:space:]')

echo "## rsync -avzP --perms --chmod=a+rwx dk@${ip}::r$(pwd) ."
