#!/usr/bin/env bash
# Local SSH tunnel to the Qwen-Image service on a rented box.
#
#   ./tunnel.sh root@136.175.252.90 49754
#   → service at http://127.0.0.1:18080
#
# Why a tunnel instead of the published port:
#   * the service can bind 127.0.0.1 inside the container, so nothing is exposed
#     to the internet even though Vast/RunPod publish a port;
#   * TLS comes from SSH, so there is no self-signed cert to trust;
#   * the endpoint is always 127.0.0.1:<local>, so the app never needs the box's
#     shifting public IP — only this process needs to know it.
#
# Env: SSH_KEY (default ~/.ssh/id_ed25519), LOCAL_PORT, REMOTE_PORT.
set -uo pipefail

TARGET=${1:?usage: tunnel.sh user@host ssh_port [local_port] [remote_port]}
SSH_PORT=${2:?usage: tunnel.sh user@host ssh_port [local_port] [remote_port]}
LOCAL_PORT=${3:-${LOCAL_PORT:-18080}}
REMOTE_PORT=${4:-${REMOTE_PORT:-10100}}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519}

echo "tunnel: 127.0.0.1:$LOCAL_PORT -> $TARGET:$REMOTE_PORT (ssh port $SSH_PORT)"
exec ssh -N -T \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile="${KNOWN_HOSTS:-/dev/null}" \
  -L "$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" \
  -i "$SSH_KEY" -p "$SSH_PORT" "$TARGET"
