#!/bin/bash
set -euo pipefail
cd /root/AKCMamba_YOLO/code
source /root/AKCMamba_YOLO/venv/bin/activate
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
echo "===== balanced train start $(date) ====="
echo "python=$(command -v python)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python train_balanced.py \
  --data ultralytics/cfg/datasets/tlfo_balanced.yaml \
  --model yolov8s.pt \
  --epochs 150 \
  --imgsz 640 \
  --batch 32 \
  --workers 8 \
  --device 0 \
  --project runs/detect \
  --name tlfo_yolov8s_balanced \
  --patience 40 \
  --seed 0
echo "===== balanced train end $(date) exit=$? ====="
