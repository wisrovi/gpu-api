#!/bin/bash
set -e

CLUSTER_NAME="gpu-cluster"

kind delete cluster --name=$CLUSTER_NAME 2>/dev/null || true

cat > /tmp/nvkind-2-workers-values.yaml << 'EOF'
workers:
- devices: 0
- devices: 0
EOF

nvkind cluster create \
  --name=$CLUSTER_NAME \
  --config-template=$(ls $HOME/go/pkg/mod/github.com/\!n\!v\!i\!d\!i\!a/nvkind*/examples/explicit-gpus-per-worker.yaml 2>/dev/null | head -1) \
  --config-values=/tmp/nvkind-2-workers-values.yaml
