import pathlib
import subprocess
import sys

import torch


def main():
    version = "2.9.0a0+50eac811a6.nv25.09"
    if torch.__version__ != version:
        raise SystemExit(f"Expected {version}, found {torch.__version__}")
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"
    ], text=True).strip()
    if processes:
        raise SystemExit("GPU compute processes detected; coordinate exclusive access. Nothing was killed.")
    if torch.cuda.device_count() != 8:
        raise SystemExit("Exactly eight visible GPUs are required")
    root = pathlib.Path("/assets")
    required = [
        root / "official-b8-20261113/ActionDiT_cpu_bf16_alpha.pt",
        root / "prepared/dataset_stats.json",
        root / "data/text_embeds_cache/libero-full",
        root / "official-b8-20261113/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors",
    ]
    model_root = root / "official-b8-20261113/models/Wan-AI/Wan2.2-TI2V-5B"
    required.extend(model_root / f"diffusion_pytorch_model-{index:05d}-of-00003.safetensors" for index in range(1, 4))
    required.append(model_root / "diffusion_pytorch_model.safetensors.index.json")
    for name in ["spatial", "object", "goal", "10"]:
        required.append(root / f"data/libero_mujoco3.3.2/libero_{name}_no_noops_lerobot/meta/info.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing assets:\n" + "\n".join(missing))
    print("Preflight passed (existence check only, not asset equivalence verification).")


if __name__ == "__main__":
    main()
