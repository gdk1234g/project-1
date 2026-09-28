# 历史实验文档：发布整理副本

> 本文保留原始记录的时间顺序与历史判断，并非当前执行计划。旧判断可能被后续实验更新；机器可读指标与每次运行的 history 分开保留。已清理本地路径、样本标识、图片引用和外部数据适配段落。本文提及的权重、归档、清单与原图均不随副本提供。

# 已完成实验记录


> 本文件用于留存已经完成的训练与消融结论。删除无效模型权重后，仍可通过本文件、各实验目录中的 `experiment.json`、`history.csv` 与 `train.log` 复核过程。


## 统一实验条件


- 任务：ATLDSD 六类苹果叶病害语义分割。

- 模型：自定义 U-Net。

- 图像尺寸：256 × 256。

- 固定划分：训练集 1,148 张、验证集 246 张、测试集 247 张；早期消融使用 seed 42，后续配对复验使用相同固定清单并分别设置 seed 43。

- 原则：消融实验只在验证集上选择方案，避免反复查看测试集。


## 基线：加权 Cross Entropy + Dice


- 实验名：`baseline_256_gpu`

- 记录位置：`运行数据第一轮/`

- 数据增强：`none`

- 损失：`ce_dice`

- 批大小 / 轮数：8 / 50

- 最佳验证结果：mIoU **0.7326**，mDice **0.8312**，最佳轮次 50。

- 测试集结果：mIoU **0.7304**，mDice **0.8315**。

- 结论：作为当前可信主基线与后续所有实验的对照，权重必须保留。


## 消融一：强数据增强


- 实验名：`augment_strong_256_gpu`

- 记录位置：`runs/augment_strong_256_gpu/`

- 相对基线的唯一改变：数据增强从 `none` 变为 `strong`；损失仍为 `ce_dice`。

- 最佳验证结果：mIoU **0.7262**，mDice **0.8269**，最佳轮次 45。

- 相对基线：mIoU **-0.0064**，mDice **-0.0042**。

- 结论：当前强增强策略没有带来收益，后续默认保持无增强。模型权重可删除；保留配置、历史 CSV 与日志作为消融证据。


## 消融二：纯 Tversky 损失


- 实验名：`loss_tversky_256_gpu`

- 记录位置：`runs/loss_tversky_256_gpu/`

- 相对基线的唯一改变：损失从 `ce_dice` 变为 `tversky`；数据增强保持 `none`。

- 训练状态：第 15 轮触发 Early Stopping，最佳轮次为 5。

- 最佳验证结果：mIoU **0.4537**，mDice **0.5736**。

- 相对基线：mIoU **-0.2789**，mDice **-0.2576**。

- 结论：纯 Tversky 丢失了加权交叉熵提供的稳定类别监督，不适合直接替换基线损失。模型权重可删除；保留配置、历史 CSV 与日志作为失败实验的可复核证据。


## 消融三：加权 CE + Dice + Tversky 混合损失


- 实验名：`mix_ce_dice_tversky_256_gpu`

- 原始结果位置：`训练后结果/mix_ce_dice_tversky_256_gpu/`

- 相对基线的唯一改变：在加权 `ce_dice` 基础上加入 Tversky；数据增强保持 `none`。

- 损失参数：`加权 CE + Dice + 0.5 × Tversky`，其中 Tversky `alpha=0.3`、`beta=0.7`。

- 批大小 / 轮数：8 / 50；未触发 Early Stopping。

- 最佳验证结果：mIoU **0.7219**，mDice **0.8236**，最佳轮次 41。

- 相对基线：mIoU **-0.0107**，mDice **-0.0075**。

- 测试集：**未评估**。由于验证集没有超过基线，本轮不进入测试集，避免产生选择偏差。

- 结论：混合损失比纯 Tversky 稳定得多，但 `0.5` 的 Tversky 权重仍降低了整体指标，不能替代当前基线。


## 基线复核：CE + Dice（seed 42）


- 实验名：`baseline_recheck_256_gpu`

- 记录位置：`baseline_recheck_256_gpu/`

- 参数：256、无增强、50 轮、batch size 8、动态类别权重。

- 最佳验证结果：训练记录 mIoU **0.7257**、mDice **0.8262**，最佳轮次 42。

- 文件：`best_unet.pth`、`history.csv`、`experiment.json`、`train.log`。


## 消融四：CE + Dice + 0.2 × Tversky（seed 42）


- 实验名：`mix_ce_dice_tversky_w02_256_gpu`

- 记录位置：`mix_ce_dice_tversky_w02_256_gpu/`

- 参数：Tversky weight 0.2、alpha 0.3、beta 0.7；其余训练参数与基线复核相同。

- 最佳验证结果：训练记录 mIoU **0.7272**、mDice **0.8279**，最佳轮次 41。

- 文件包含权重、历史 CSV、实验摘要；没有 `train.log`，服务器 split 快照/hash 也没有随结果保存。

- 在同一份本地验证集复算：基线 mIoU/mDice **0.7258/0.8262**；混合 0.2 为 **0.7272/0.8279**。整体仅差约 0.15 个百分点。

- 灰斑 IoU/Recall：基线 **0.5759/0.7200**，混合 0.2 **0.5613/0.6968**；混合方案没有改善灰斑漏检。交链孢叶斑和褐斑有所改善，锈病和灰斑下降。

- 结论：单次结果不足以判定混合损失稳定胜出。类别权重相同、样本数相同，但缺少服务器 split hash，无法确认旧训练逐样本使用相同划分。


## 配对复验：CE + Dice 对比 CE + Dice + 0.2 × Tversky（seed 43）


- 实验名：`baseline_seed43` 与 `mix_w02_seed43`

- 记录位置：`baseline_seed43/`、`mix_w02_seed43/`

- 训练状态：两组均完成 50 轮，最佳轮次均为 49；每组均保留 `best_unet.pth`、`history.csv`、`experiment.json`、`train.log`、配置快照、环境快照和三份 split 快照。

- 配对条件：seed 43、图像尺寸 256、无增强、batch size 8、学习率 0.001、weight decay 0.0001、最多 50 轮、early stopping patience 10；两组 train/val 样本数分别为 1,148/246，动态类别权重相同。配置除损失及其参数、评估权重输出路径外一致。

- 数据根目录 / 仓库：服务器配置记录为 `[LOCAL_PATH]` / `[LOCAL_PATH]`。

- split SHA256（两组相同）：train `5e7a19f2add7edd20f16df3c931403daeb126d0a7a58a[SAMPLE_ID]d8b10e61115de`；val `9e048b32771837335e29af2912612fc518bd6f5f1544092bef704b733187b6a4`；test `4ebff89b96cc6721cbe91d26f680981c80fffb0438dce4b10a2cc1a6057a21d1`。保存的清单副本与本地 `splits/` 逐字节一致。

- 环境快照：两组 `environment.txt` SHA256 相同，为 `2adc829c839a6be75a5db1fcdbe65ddc35975bfa4b8e45600e0f2b09380e45c2`。

- CE + Dice 基线最佳验证结果：mIoU **0.7323**，mDice **0.8325**，第 49 轮。

- 混合损失最佳验证结果：mIoU **0.7364**，mDice **0.8359**，第 49 轮；损失为加权 CE + Dice + `0.2 × Tversky`，alpha=0.3、beta=0.7。

- 相对 seed 43 基线：mIoU **+0.0041**（约 +0.41 个百分点），mDice **+0.0034**（约 +0.34 个百分点）。

- 测试集：本轮未用于选模；不据此结果运行测试集挑选方案。

- 结论：在已确认逐字节相同的 train/val/test 清单、相同 seed 和训练条件下，seed43 混合版本次验证指标略高；单个配对 seed 的提升较小，尚不能证明稳定优于 CE + Dice。seed44 配对结果已完成，见下一节。


## 配对复验：CE + Dice 对比 CE + Dice + 0.2 × Tversky（seed 44）


- 实验名：`baseline_seed44` 与 `mix_w02_seed44`；配置为 `configs/baseline_seed44.json` 与 `configs/mix_w02_seed44.json`。

- 当前本地留存：两组均有 `best_unet.pth`、`history.csv`、`experiment.json`。基线历史含 50 轮，混合版历史含 47 轮；按验证 mIoU 取最佳轮次，基线为第 43 轮，混合版为第 37 轮。

- 最佳验证结果：基线 mIoU **0.7379**、mDice **0.8361**；混合版 mIoU **0.7225**、mDice **0.8250**。

- seed 44 配对差值（混合版减基线）：mIoU **-0.0155**（约 -1.55 个百分点），mDice **-0.0111**（约 -1.11 个百分点）。

- 配置记录显示两组均为 256、无增强、batch size 8、50 轮上限、train/val 1,148/246，动态类别权重相同；混合版使用 `tversky_weight=0.2`、`alpha=0.3`、`beta=0.7`。

- **运行前核对与复现证据：**用户确认在启动 seed44 两组训练前已核对参数一致；本地两份配置及实验摘要也相互吻合，除随机种子对应的配对运行外，实验因素仅为损失函数及其参数。服务器没有训练日志，本地目录也没有配置/环境快照、`split_sha256.txt` 或 train/val/test 清单副本。因此参数一致性有运行前人工核对和配置记录支持，但目前无法事后复核训练过程、停止原因，以及训练时数据划分/环境的哈希。不要用本地当前 split 哈希替代训练时的哈希。

- 结论：seed43 中混合版略高，seed44 中混合版明显较低；方向不一致。当前不能判定 0.2 混合损失稳定优于基线，亦不应凭 seed43 单次结果选它。


## 下一轮方向


## 配对复验：CE + Dice 对比 CE + Dice + 0.2 × Tversky（seed 45）


- 实验名：基线 `baseline_seed45_new`、混合版 `mix_w02_seed45`。当前本地记录目录为 `baseline_seed45_new/`、`mix_w02_seed45/`；前者目录名与配置/原计划的 `baseline_seed45` 不同，分析按 `experiment.json` 中的损失和 seed 45 参数识别。

- 两组均记录 50 轮；按验证 mIoU 选出的最佳轮次：基线第 46 轮，混合版第 44 轮。

- 最佳验证结果：基线 mIoU **0.7347**、mDice **0.8348**；混合版 mIoU **0.7315**、mDice **0.8313**。

- seed45 配对差值（混合版减基线）：mIoU **-0.0033**（约 -0.33 个百分点），mDice **-0.0035**（约 -0.35 个百分点）。

- **参数核对：**用户确认在运行前已核对 seed45 两组基础训练参数一致；本地 `experiment.json` 也显示两组图像尺寸、增强方式、样本数及动态类别权重相同，混合版额外使用 Tversky weight 0.2、alpha 0.3、beta 0.7。用户说明服务器数据路径为 `[LOCAL_PATH]`；两份 seed45 配置的本地工作副本也已改为该路径。归档缺项不代表参数未核对或参数不一致。

- **归档核查：**两个目录均有约 31 MB 的 `best_unet.pth`、50 轮 `history.csv` 和 `experiment.json`。但本机收到的补充文件并非完整复现快照：`train.log` 只有 `Training log` 一行；`split_sha256.txt` 是 `dataset split hash placeholder`；所谓 train/val/test JSON 只有几十字节的指标摘要，不是 split 清单副本；`config_snapshot.json` 只有 name/epochs/seed，且混合版目录中也是基线名；`environment.txt` 只有 Python/GPU/Torch 简要信息；未见 `git_commit.txt` 或源码快照。这是归档完整性问题，不影响用户已做的运行前参数核对；若服务器原始运行目录仍保留真实快照，可补档，但不需要因此重跑或重复核对参数。

- 结论：seed45 混合版低于基线。结合前三个配对 seed，mIoU 差值依次为 **+0.0041、-0.0155、-0.0033**，平均 **-0.0049**；mDice 差值为 **+0.0034、-0.0111、-0.0035**，平均 **-0.0037**。三个 seed 中有两个支持基线，且平均差值也偏向基线；目前没有证据支持把 0.2 混合损失作为整体指标更好的默认方案。


## 下一轮方向


当前建议先停止继续训练 seed46 或扫描 Tversky 系数，暂以 CE + Dice 作为整体分割指标的候选方案。seed45 归档缺项不阻塞基于权重和 history 的当前判断，也无需重跑实验。若研究目标包括特定病害类别的召回率，下一步在固定验证集上对 seed43–45 的最佳权重补做逐类别指标对比，使用新增的 `scripts/evaluate_validation.py`，操作见 `reports/日志完整留存和验证集逐类别评估.md`；不要使用只评估测试集的 `scripts/evaluate_baseline.py` 做选型。锁定模型后再对最终方案进行一次测试集评估，不用测试集反复选参。未来训练使用 `scripts/run_training_logged.py`，自动记录完整 stdout/stderr 和运行快照。



