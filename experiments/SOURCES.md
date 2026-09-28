# 数据、方法与实验文件来源

## 1. 原始数据集：ATLDSD

| 项目 | 来源信息 |
|---|---|
| 中文名称 | 苹果树叶病分割数据集 |
| 英文名称 | Apple Tree Leaf Disease Segmentation Dataset（ATLDSD） |
| 数据作者 | 冯景泽、赵晓飞 |
| 发布平台与引用版本 | ScienceDB（科学数据银行），2022，V1 |
| DOI | [10.11922/sciencedb.01627](https://doi.org/10.11922/sciencedb.01627) |
| CSTR | [31253.11.sciencedb.01627](https://cstr.cn/31253.11.sciencedb.01627) |
| 数据记录页 | [ScienceDB 原始记录](https://www.scidb.cn/en/detail?dataSetId=0e1f57004db842f99668d82183afd578) |
| 数据许可 | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |

推荐引用：冯景泽，赵晓飞．苹果树叶病分割数据集[DS/OL]．V1．ScienceDB，2022．DOI：10.11922/sciencedb.01627．

作者、版本与许可依据数据记录信息及此前提供的许可页面截图整理；截图不随仓库发布。原始图像与标签属于数据集材料，本项目使用它们开展六类语义分割研究，不将它们声明为自己的原创数据。使用者应到原始记录页获取数据并核对对应版本的使用条件；本仓库不是数据下载镜像。

## 2. 本仓库实验文件从哪里来

| 公开文件 | 实际来源及整理方式 |
|---|---|
| `configs/*.json` | 本项目编写并留存的历史参数文件，不是数据作者发布的官方训练参数。仅将数据、权重和划分位置改成通用路径；数值参数保留。 |
| `runs/*/experiment.json` | 对应训练运行生成的参数记录；提取实验名、损失、批大小、样本数量等允许公开字段。 |
| `runs/*/history.csv` | 对应训练运行逐轮记录的训练损失、验证损失和分割指标；保留数字字符串，统一编码及换行。 |
| `runs/*/train*.log` | 真实日志的训练数值行摘录；移除启动命令、路径及非训练输出。不是完整原始日志。 |
| `runs/*/config_snapshot.json` | 运行目录中留存的配置快照；泛化路径，不补造缺失快照。其留存情况不等于独立验证全部运行过程。 |
| `runs/*/unverified_config_copy.json` | 旧目录中的配置副本；原日志存在占位内容，不能确认该副本完整反映实际训练。单独标为未核实。 |
| `runs/*/package_versions.json` | 从对应留存环境信息中提取的包名与版本；不包含完整环境变量或机器信息。 |
| `metrics/*/metrics.json`、`per_class_metrics.csv` | 本项目评估程序生成的聚合验证/测试报告；保留混淆矩阵与逐类别指标，去除本机路径。不是数据集官方基准成绩。 |
| `training_summary.csv`、`run_index.json` | 从留存 history 计算各运行的最佳验证 mIoU 轮次及同轮 Dice，并登记材料缺项。 |
| `configuration_summary.csv` | 汇总每组运行记录与其留存配置；缺失或未核实的字段明确标注，不根据目录名补写。 |
| `historical_configuration_summary.csv` | 汇总 15 份历史参数文件；表示文件中记录的计划参数，不自动证明某次训练实际使用。 |
| `notes/*.md` | 本项目过程文档的清理副本；保留历史判断，去除敏感路径、样本引用和外部数据适配段落。 |
| `EXPERIMENT_SUMMARY.md`、Word 总报告 | 根据上述发布材料重新组织的汇总；清楚区分训练指标、后续复评和来源缺项。 |
| `release_manifest.json` | 整理完成后计算的文件 SHA256，用于检查本次发布副本是否改变，不是原始数据集文件清单。 |

原始运行目录与压缩归档中的文字材料是此次整理的输入。数据清单、图片、标注、权重和原始压缩包没有复制到公开目录。相同副本合并，内容不同的历史运行分别保留。详细原始路径映射仅留在本机审计文件中。

## 3. 网络方法来源

Ronneberger, O.; Fischer, P.; Brox, T. U-Net: Convolutional Networks for Biomedical Image Segmentation. 2015. [原论文](https://arxiv.org/abs/1505.04597)。

这一引用说明架构方法来源，不等于完成所有实现代码的版权来源核查。数据许可、论文许可与本项目源码的许可分别说明；本仓库源码许可尚待来源核实后确定。

## 4. 外部数据尝试：PlantSeg

项目曾尝试 ATLDSD 与 PlantSeg 苹果锈病样本混合适配，未获得满足本项目评估约束的候选，最终保留 ATLDSD 基线。公开材料仅保留自行撰写的简短过程概述。

- 引用来源：Tianqi Wei，PlantSeg: A Large-Scale In-the-wild Dataset for Plant Disease Segmentation，Zenodo，v3。
- 版本 DOI：[10.5281/zenodo.13924591](https://doi.org/10.5281/zenodo.13924591)。
- 当时使用版本的许可依据此前核实页面为 [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)。不同版本的许可分别核对，不能用后续版本的许可替代历史版本。
- 不提供此次尝试的图片、标注、转换标签、权重或实验附件，也不暗示数据作者为本项目背书。

## 5. 使用和解读边界

本项目是学习研究与个人作品展示，未提供商业化服务。原始数据的使用仍按其对应许可执行。聚合实验分数是本项目的计算结果，不是诊断结论或数据作者发布的成绩。
没有公开逐样本划分、权重和原训练时的完整源码快照，不能承诺凭本仓库逐字节重现历史实验。旧测试集曾被查看；不能把它当成后续完全未触碰的测试集。
