# ATLDSD 实验汇总

本表从留存的 history.csv 计算；最佳轮次按验证 mIoU 选择，同一行 Dice 是该轮的 Dice。分数没有因发布整理而调整。

## 来源说明

原始数据引用：冯景泽、赵晓飞．苹果树叶病分割数据集，ScienceDB，2022，V1，DOI：[10.11922/sciencedb.01627](https://doi.org/10.11922/sciencedb.01627)，许可：[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。本文件中的参数、分数和结论来自本项目自己的训练与评估记录，不是数据作者发布的基准成绩。文件生成方式、版本和清理规则见 [SOURCES.md](SOURCES.md)。

完整配置与逐类别结果另见 [Word 总报告](ATLDSD_实验结果与配置总报告.docx)。

## 训练记录

| 实验 | 记录轮数 | 最佳轮次 | 验证 mIoU | 同轮 mDice | 有数值训练日志 | 配置情况 |
|---|---:|---:|---:|---:|---|---|
| [augment_strong_256_gpu](runs/augment_strong_256_gpu/publication_notes.json) | 50 | 45 | 0.726206 | 0.826933 | 有 | 缺少运行快照 |
| [baseline_256_gpu](runs/baseline_256_gpu/publication_notes.json) | 50 | 50 | 0.732624 | 0.831180 | 无 | 缺少运行快照 |
| [baseline_recheck_256_gpu](runs/baseline_recheck_256_gpu/publication_notes.json) | 50 | 42 | 0.725732 | 0.826157 | 有 | 缺少运行快照 |
| [baseline_seed43](runs/baseline_seed43/publication_notes.json) | 50 | 49 | 0.732326 | 0.832487 | 有 | 有留存快照 |
| [baseline_seed44](runs/baseline_seed44/publication_notes.json) | 50 | 43 | 0.737914 | 0.836060 | 无 | 缺少运行快照 |
| [baseline_seed45_complete](runs/baseline_seed45_complete/publication_notes.json) | 50 | 46 | 0.725204 | 0.827070 | 有 | 有留存快照 |
| [baseline_seed45_new](runs/baseline_seed45_new/publication_notes.json) | 50 | 46 | 0.734720 | 0.834848 | 无 | 有未核实副本 |
| [baseline_seed46_res256](runs/baseline_seed46_res256/publication_notes.json) | 50 | 40 | 0.735725 | 0.835773 | 有 | 有留存快照 |
| [baseline_seed46_res384](runs/baseline_seed46_res384/publication_notes.json) | 50 | 50 | 0.722093 | 0.825235 | 有 | 有留存快照 |
| [loss_tversky_256_gpu](runs/loss_tversky_256_gpu/publication_notes.json) | 15 | 5 | 0.453741 | 0.573615 | 有 | 缺少运行快照 |
| [mix_ce_dice_tversky_256_gpu](runs/mix_ce_dice_tversky_256_gpu/publication_notes.json) | 50 | 41 | 0.721933 | 0.823633 | 有 | 缺少运行快照 |
| [mix_ce_dice_tversky_w02_256_gpu](runs/mix_ce_dice_tversky_w02_256_gpu/publication_notes.json) | 50 | 41 | 0.727212 | 0.827902 | 无 | 缺少运行快照 |
| [mix_w02_seed43](runs/mix_w02_seed43/publication_notes.json) | 50 | 49 | 0.736384 | 0.835932 | 有 | 有留存快照 |
| [mix_w02_seed44](runs/mix_w02_seed44/publication_notes.json) | 47 | 37 | 0.722455 | 0.824966 | 无 | 缺少运行快照 |
| [mix_w02_seed45](runs/mix_w02_seed45/publication_notes.json) | 50 | 44 | 0.731469 | 0.831349 | 无 | 有未核实副本 |
| [mix_w02_seed45_complete](runs/mix_w02_seed45_complete/publication_notes.json) | 50 | 42 | 0.722057 | 0.823974 | 有 | 有留存快照 |

## 阶段结论

- seed43–45 的 CE+Dice 与 CE+Dice+0.2×Tversky 配对结果方向不一致，保留 CE+Dice 基线。seed45 的旧运行与 complete 补训分开保留。
- seed46 256/384 配对验证中，256 的 mIoU 为 0.735725，384 为 0.722093；本轮保留 256。这个结果只说明本轮配置与固定验证集上的差异。
- 当前演示候选为 seed46 256、第 40 轮。seed44 的历史验证 mIoU 更高，不能称 seed46 为所有实验的唯一最佳。
- 早期部分实验没有完整日志、环境或原始训练配置；占位文字没有作为训练证据发布。当前整理日志仅保留训练数值行。

## 独立评估报告

训练过程中记录的验证分数与后续 CPU/CUDA 复评分别保存，不以四舍五入后的相同分数合并不同报告。没有对应训练 history 的旧微调报告单独注明。

| 报告 | 集合 | mIoU | mDice | checkpoint轮次 | 对应训练记录 |
|---|---|---:|---:|---:|---|
| [legacy_finetune_evaluation](metrics/legacy_finetune_evaluation/metrics.json) | test | 0.742299 | 0.841165 | 53 | 未确认 |
| [mix_w02_seed45__seed45_complete_validation__baseline](metrics/mix_w02_seed45__seed45_complete_validation__baseline/metrics.json) | validation | 0.725204 | 0.827070 | 46 | baseline_seed45_complete |
| [mix_w02_seed45__seed45_complete_validation__mix_w02](metrics/mix_w02_seed45__seed45_complete_validation__mix_w02/metrics.json) | validation | 0.722057 | 0.823974 | 42 | mix_w02_seed45_complete |
| [paired_validation_seed43__baseline](metrics/paired_validation_seed43__baseline/metrics.json) | validation | 0.732326 | 0.832487 | 49 | baseline_seed43 |
| [paired_validation_seed43__mix_w02](metrics/paired_validation_seed43__mix_w02/metrics.json) | validation | 0.736384 | 0.835932 | 49 | mix_w02_seed43 |
| [paired_validation_seed44__baseline](metrics/paired_validation_seed44__baseline/metrics.json) | validation | 0.737914 | 0.836060 | 43 | baseline_seed44 |
| [paired_validation_seed44__mix_w02](metrics/paired_validation_seed44__mix_w02/metrics.json) | validation | 0.722455 | 0.824966 | 37 | mix_w02_seed44 |
| [final_test_seed45_baseline](metrics/final_test_seed45_baseline/metrics.json) | test | 0.714559 | 0.818135 | 46 | baseline_seed45_complete |
| [seed46_resolution_validation__res256](metrics/seed46_resolution_validation__res256/metrics.json) | validation | 0.735725 | 0.835773 | 40 | baseline_seed46_res256 |
| [seed46_resolution_validation__res384](metrics/seed46_resolution_validation__res384/metrics.json) | validation | 0.722093 | 0.825235 | 50 | baseline_seed46_res384 |
| [baseline_256_gpu_test](metrics/baseline_256_gpu_test/metrics.json) | test | 0.730408 | 0.831453 | 50 | baseline_256_gpu |
| [seed43_44_classwise__baseline_seed43](metrics/seed43_44_classwise__baseline_seed43/metrics.json) | validation | 0.732335 | 0.832494 | 49 | baseline_seed43 |
| [seed43_44_classwise__baseline_seed44](metrics/seed43_44_classwise__baseline_seed44/metrics.json) | validation | 0.737918 | 0.836063 | 43 | baseline_seed44 |
| [seed43_44_classwise__mix_w02_seed43](metrics/seed43_44_classwise__mix_w02_seed43/metrics.json) | validation | 0.736367 | 0.835919 | 49 | mix_w02_seed43 |
| [seed43_44_classwise__mix_w02_seed44](metrics/seed43_44_classwise__mix_w02_seed44/metrics.json) | validation | 0.722434 | 0.824948 | 37 | mix_w02_seed44 |

## 历史文档

- [seed43–45 逐类别验证汇总](notes/seed43_45_validation_133c05dd.md)。历史快照或专题复核。
- [较早版本的实验历史记录](notes/experiment_record_c61a30ab.md)。历史快照或专题复核。
- [ATLDSD 基线阶段总结](notes/baseline_summary_ed51c27e.md)。历史快照或专题复核。
- [较早版本的实验历史记录](notes/experiment_record_0bcf814b.md)。历史快照或专题复核。
- [标注复核与模型局限](notes/annotation_review_limits_62a20fd2.md)。历史快照或专题复核。
- [seed46 分辨率实验结果复核](notes/seed46_resolution_review_8bd73d7d.md)。历史快照或专题复核。
- [主实验历史记录](notes/experiment_record_eec4f2de.md)。当前汇总记录。
- [较早版本的实验历史记录](notes/experiment_record_7f5652b8.md)。历史快照或专题复核。

完整原精度数值见 JSON/CSV。数据、逐样本划分与权重不随发布副本提供；仅凭公开材料不能保证精确重现原训练。

## 补充：外部数据混合训练尝试

项目还尝试过将 ATLDSD 与 PlantSeg 的苹果锈病样本进行混合适配训练。部分设置出现误检增加和 ATLDSD 性能下降，调整后仍未获得满足本项目评估约束的候选，因此最终保留 ATLDSD 基线模型，未将混合训练作为改进成果。

PlantSeg 来源：[Tianqi Wei，PlantSeg，Zenodo v3](https://doi.org/10.5281/zenodo.13924591)。当时使用版本的许可已核实为 [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)。为谨慎处理该版本许可，本仓库仅保留这段自行撰写的尝试概述，不提供相关图片、标注、转换标签、权重或实验附件。
