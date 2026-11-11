#!/bin/bash
# 双机同步助手：本地 Slipstream <-> lab42 (zoo@192.168.202.42:/home/zoo/rino_we/808-Scale-AD)
# 架构：git 为代码真值（lab42 remote, receive.denyCurrentBranch=updateInstead 已配）；
#       artifacts/.scratch 为本地单向镜像（远端是证据副本，不反向）。
# 用法：
#   scripts/sync_lab42.sh            # 常规：rsync artifacts/.scratch + git push lab42 master
#   scripts/sync_lab42.sh --pull     # 反向：拉取远端 master（远端有提交时）
#   scripts/sync_lab42.sh --full     # 全量镜像（含 --delete，慎用：会删远端已消失文件）
set -e
cd "$(dirname "$0")/.."
REMOTE_HOST="zoo@192.168.202.42"
REMOTE_DIR="/home/zoo/rino_we/808-Scale-AD/Slipstream"
RSYNC_OPTS="-a --info=stats0"
case "${1:-push}" in
  --pull)
    git fetch lab42 && git merge --ff-only lab42/master && echo "[sync] pulled lab42/master"
    ;;
  --full) RSYNC_OPTS="$RSYNC_OPTS --delete" ;&
  push|"")
    rsync $RSYNC_OPTS --exclude='.venv/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='build/' \
      artifacts/ .scratch/ "$REMOTE_HOST:$REMOTE_DIR/" 2>/dev/null || \
    rsync $RSYNC_OPTS artifacts/ "$REMOTE_HOST:$REMOTE_DIR/artifacts/" && \
    rsync $RSYNC_OPTS .scratch/ "$REMOTE_HOST:$REMOTE_DIR/.scratch/"
    git push lab42 master && echo "[sync] pushed master -> lab42"
    ;;
esac
