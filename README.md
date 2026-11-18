# FastWAM H20 — NGC 25.09 benchmark delivery

本仓库整理一次 H20 八卡 FastWAM 100-step 实测的代码、环境说明与脱敏证据。它是训练性能审查包，不是 CUDA/MUSA 精度验收证书，也不包含 S5000 实现。

## 上游来源与修改说明

本仓库基于 [FastWAM](https://github.com/yuantianyuan01/FastWAM)，原项目由 **The FastWAM Authors** 开发，采用 **MIT License**。原项目版权及许可声明保留于 [LICENSE](LICENSE) 和 [FastWAM/LICENSE](FastWAM/LICENSE)。

上游基线 commit：

`7faa71108368fbb3b6885649f112af607427a2d4`

本仓库是用于 NVIDIA H20 / NGC PyTorch 25.09 环境的训练性能测试与复现材料整理，**不是 FastWAM 官方仓库**，不主张对上游项目的原创权利。

本仓库包含环境适配、训练计时与运行记录相关修改，以及环境构建说明、启动脚本和测试结果。具体源码变更清单及校验信息见 [provenance/source.json](provenance/source.json)；上游项目说明见 [FastWAM/README.md](FastWAM/README.md)。

本仓库的测试结果与结论由本次测试提供方负责，不代表上游作者的认可或背书。模型权重、数据集及第三方组件应分别遵守其适用许可证，不因本仓库保留 MIT License 而自动获得分发许可。

## 已有结果

| 项目 | 记录值 |
|---|---:|
| GPU | 8 × NVIDIA H20-3e |
| micro batch / accumulation / GBS | 8 / 1 / 64 |
| 精度 / DeepSpeed | BF16 / ZeRO-1，无 offload |
| compile / gradient checkpointing | false / false |
| 总步数 / 预热 / 测量 | 100 / 10 / 90 |
| 有效样本 / 时间 | 5760 / 130.890100686 s |
| 吞吐 / optimizer step | 44.00638375 samples/s / 1.454334452 s |

结果位于 results/h20-ngc2509-100/。本包不改变产生这些结果的训练/计时代码。该轮在第11步取batch前同步开始，在第100步更新后同步结束，不包含最后一步更新后的指标gather及日志；其他窗口内开销包含在内。结果为单次测量，未证明跨平台精度对齐或稳定领先比例。

## 内容

- FastWAM/：服务器工作树快照，包含本轮实际修改后的 src、scripts、configs、上游README和许可证。不包含无关第三方评估环境、数据、checkpoint或旧Git历史。
- environment/：NGC版本、额外依赖、运行时软件清单。
- scripts/：新增交付启动及证据验证工具；不是历史运行的原始启动器。
- results/h20-ngc2509-100/：脱敏后的原始记录；command.sh仅供审查，不建议直接执行。
- provenance/：上游commit、本次源文件哈希、脱敏规则及打包清单。
- docs/ASSETS.md：必须另行准备的外部资产。
- docs/LIMITATIONS.md：复现及对比边界。

## 快速验证已有结果（不需要GPU）

    python3 scripts/verify_results.py results/h20-ngc2509-100
    sha256sum -c provenance/SHA256SUMS

## 准备和运行

先阅读 docs/ASSETS.md。权重、数据和缓存不在Git仓库中，必须取得许可后准备；目前尚未在全新机器上验证完整重建，不承诺clone后自动得到相同吞吐。

    docker build -t fastwam-h20:ngc2509 .
    export ASSET_ROOT=/absolute/path/to/fastwam-assets
    bash scripts/run_container.sh

ASSET_ROOT须是宿主机上的绝对目录。脚本只读挂载资产、在仓库runs/下写新结果。训练前检查八张卡和GPU进程；若有其他计算进程则退出，不杀进程，不自动降卡数。仍需管理员协调独占，检查不能避免启动时的资源竞争。

构建保留NGC自带torch/torchvision，通过额外依赖文件补齐环境。**不要按照FastWAM/README.md中的旧CUDA安装步骤覆盖NGC的torch，也不要直接pip install -e FastWAM让旧pyproject依赖替换本轮版本。** 当前入口通过PYTHONPATH加载源码，无需安装项目。

新启动器只改交付路径和可配置端口，保留8卡、batch8、100步等历史参数及原计时器，不用于解决两平台计时一致性问题。所有新增运行进入runs/，不会覆盖历史results/。

## 发布与授权

默认建议私有GitHub仓库。上游许可证见LICENSE与FastWAM/LICENSE；公开公司修改、模型及数据前仍需获得授权。本包不包含密码、Token或SSH配置，不保留服务器Git历史。自动脱敏/扫描不能代替人工合规审查。

GitHub上传步骤见 docs/PUBLISH.md。请交付固定commit，明确这是一份阶段性实测记录。
