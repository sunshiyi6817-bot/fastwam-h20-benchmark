#!/usr/bin/env bash
set -euo pipefail
cd /aligned/FastWAM
export PYTHONPATH=/aligned/FastWAM/src
export DIFFSYNTH_MODEL_BASE_PATH=/assets/official-b8-20261113/models
export DIFFSYNTH_SKIP_DOWNLOAD=true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export FASTWAM_PHASE=formal
export FASTWAM_RUN_DIR="/aligned/runs/h20_ngc2509_$(date -u +%Y%m%dT%H%M%S)_$$"
python /aligned/scripts/preflight.py
mkdir -p "$FASTWAM_RUN_DIR"
nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv > "$FASTWAM_RUN_DIR/gpu-before.csv"
set +e
accelerate launch   --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml   --num_processes 8 --main_process_port "${MAIN_PROCESS_PORT:-29519}"   scripts/train.py hydra/job_logging=disabled hydra/hydra_logging=disabled   task=libero_uncond_2cam224_1e-4 batch_size=8 gradient_accumulation_steps=1   mixed_precision=bf16 max_steps=100 model.compile_training_denoise=false   model.mot_checkpoint_mixed_attn=false resume=null seed=42 num_workers=8   log_every=1 save_every=0 eval_every=0   model.action_dit_pretrained_path=/assets/official-b8-20261113/ActionDiT_cpu_bf16_alpha.pt   'data.train.dataset_dirs=[/assets/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot,/assets/data/libero_mujoco3.3.2/libero_object_no_noops_lerobot,/assets/data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot,/assets/data/libero_mujoco3.3.2/libero_10_no_noops_lerobot]'   data.train.text_embedding_cache_dir=/assets/data/text_embeds_cache/libero-full   +data.train.pretrained_norm_stats=/assets/prepared/dataset_stats.json   +data.train.video_backend=pyav "output_dir=$FASTWAM_RUN_DIR"   > "$FASTWAM_RUN_DIR/launcher.log" 2>&1
status=$?
set -e
printf '{"exit_code":%d,"training_started":true}
' "$status" > "$FASTWAM_RUN_DIR/status.json"
printf 'Run directory: %s
' "$FASTWAM_RUN_DIR"
if [ "$status" -ne 0 ]; then exit "$status"; fi
python /aligned/scripts/verify_results.py "$FASTWAM_RUN_DIR"
