#!/bin/bash
set -euo pipefail
cd /root/AKCMamba_YOLO/code
source /root/AKCMamba_YOLO/venv/bin/activate
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
echo "===== train start $(date) ====="
echo "python=$(which python)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python train_server.py \
  --data ultralytics/cfg/datasets/tlfo.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 32 \
  --workers 8 \
  --device 0 \
  --project runs/detect \
  --name tlfo_yolov8n \
  --patience 50 \
  --seed 0
echo "===== train end $(date) exit=$? ====="
