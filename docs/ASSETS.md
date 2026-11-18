# 外部资产准备

仓库不分发权重、视频、训练数据或缓存。下列路径相对于ASSET_ROOT，容器内统一为/assets。

    official-b8-20261113/
      ActionDiT_cpu_bf16_alpha.pt
      models/
        Wan-AI/Wan2.2-TI2V-5B/
          diffusion_pytorch_model-00001-of-00003.safetensors
          diffusion_pytorch_model-00002-of-00003.safetensors
          diffusion_pytorch_model-00003-of-00003.safetensors
          diffusion_pytorch_model.safetensors.index.json
        Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/  (tokenizer文件)
        DiffSynth-Studio/Wan-Series-Converted-Safetensors/
          Wan2.2_VAE.safetensors
    data/libero_mujoco3.3.2/
      libero_spatial_no_noops_lerobot/
      libero_object_no_noops_lerobot/
      libero_goal_no_noops_lerobot/
      libero_10_no_noops_lerobot/
    data/text_embeds_cache/libero-full/
    prepared/dataset_stats.json

模型加载器实际所需tokenizer目录及文件应以资产提供方和FastWAM代码为准。原服务器models/DiffSynth-Studio是指向/assets/models/DiffSynth-Studio的链接；可在上述目录放真实文件，或保留链接并同时提供目标目录。不能复制宿主机失效的绝对符号链接后就认为资产完整。

## 可核实的来源与步骤

- Wan模型ID：Wan-AI/Wan2.2-TI2V-5B；历史分片revision/sha256见provenance/wan-weights-reference.json。该文件来自原准备记录，本次未重新计算数十GB权重哈希。
- VAE模型ID：DiffSynth-Studio/Wan-Series-Converted-Safetensors。原运行load_text_encoder=false，但仍需满足tokenizer加载路径。
- 上游数据说明：https://huggingface.co/datasets/yuanty/LIBERO-fastwam 。该地址来自上游README，不能代替对本轮实际文件的内容核验。
- 文本缓存生成入口和ActionDiT生成脚本随FastWAM/scripts交付，先查看其--help和上游说明。

历史ActionDiT预处理调用为（从FastWAM目录执行，先配置DIFFSYNTH_MODEL_BASE_PATH）：

    python scripts/preprocess_action_dit_backbone.py \
      --model-config configs/model/fastwam.yaml \
      --output /assets/official-b8-20261113/ActionDiT_cpu_bf16_alpha.pt \
      --device cpu --dtype bfloat16 --apply-alpha-scaling true

历史ActionDiT SHA256为：ec43a97d99a8929294ee29894933ef9f48373100159871dbfc9a7cd815242f12。上述命令来自原准备记录，生成结果仍应核验，不应把任意同名公开checkpoint当作同一权重。

prepared/dataset_stats.json及libero-full文本缓存需要从本轮资产提供方取得，或按明确预处理流程重建并核验；本交付不声称已完整还原两者的生成来源。运行时统计文件的SHA256见provenance/asset-checks.json；未把数据统计内容作为公开Git资产。

## 原服务器复用

无需复制大资产：将ASSET_ROOT设置为本轮资产所在宿主目录即可。该路径由管理员提供，不写入公开仓库。代码与运行目录独立，不修改原资产。
