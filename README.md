# 苹果叶病害分割：源码与本地推理演示

这是项目已提取的基础源码，提供 U-Net 网络结构、分割损失、像素级指标、数据读取与增强、基础配置、报告生成、命令行预测及网页前后端。

## 文件

| 文件 | 用途 |
| --- | --- |
| src/model.py | U-Net 编码器、解码器与跳跃连接 |
| src/baseline_losses.py | Dice、Focal、Tversky 与组合损失 |
| src/baseline_metrics.py | 混淆矩阵、IoU、Dice、Precision、Recall、F1 等指标计算 |
| src/__init__.py | Python 包初始化 |
| src/config.py | 类别名称、颜色、输入尺寸及可配置目录 |
| src/dataset.py | 用户本地 image/label 配对、图像与类别索引掩膜读取 |
| src/baseline_dataset.py | 图像与掩膜同步增强、缩放和归一化 |
| src/reporting.py | 从调用方传入的指标生成 CSV、JSON 和混淆矩阵 |
| src/visualize.py | 类别着色、病斑面积统计与叠加图生成 |
| predict.py | 使用用户指定的权重进行单图预测 |
| demo/app.py | 文件校验、权重校验、推理及 FastAPI 接口 |
| demo/static/index.html | 图片上传与预测展示界面 |
| demo/static/app.css | 桌面和手机页面样式 |
| demo/static/app.js | 预览、请求、结果展示及下载 |
| demo/__init__.py | 网页服务包初始化 |
| demo/requirements.txt | 网页服务依赖 |
| demo/start.cmd | Windows 本地启动脚本 |
| src/experiment.py | 默认参数、随机种子及固定划分的读写校验 |
| scripts/train_baseline.py | 动态类别权重、训练、早停及 checkpoint 保存 |
| scripts/evaluate_validation.py | 使用本地固定验证集评估用户指定权重 |
| scripts/audit_dataset.py | 检查本地图片与标签、生成用户自己的划分 |
| scripts/__init__.py | 命令行脚本包初始化 |
| scripts/evaluate_baseline.py | 固定测试集的最终评估入口 |
| scripts/run_training_logged.py | 实时显示训练输出并独立保存日志 |

当前副本包括基础模块、单图推理、训练、数据检查和固定验证集评估入口。输入数据由使用者自行在本地准备，运行产物也保存在本地。数据集目录默认位于项目下的 data/ATLDSD，也可通过 ATLDSD_ROOT 环境变量指定外部目录。该路径只指示读取位置，并不提供数据文件。

## 依赖

Python 3.10 或更新版本；PyTorch、NumPy、OpenCV、Pillow 和 Matplotlib。依赖清单见 requirements.txt。使用 GPU 时，应为对应设备安装合适的 PyTorch 构建。


## 本地网页启动

先准备你有权使用的本项目六类 U-Net checkpoint。格式为字典，其 model 字段是 src/model.py 定义的 UNet（base=32）的 state_dict；输入为 256×256，使用 ImageNet 归一化及水平翻转 TTA。

本副本没有附带权重；默认查找 checkpoints/best_unet.pth，也支持 ATLDSD_WEIGHTS 指向外部文件。后端要求提供所选文件的 SHA256，且不会把一份未知权重自动认定为原项目的实验模型。

在已具备合适 PyTorch 环境的项目根目录安装依赖：

```powershell
python -m pip install -r requirements.txt
python -m pip install -r demo/requirements.txt
```

在同一个 PowerShell 终端设置权重与校验值，再启动：

```powershell
$env:ATLDSD_WEIGHTS = (Resolve-Path './checkpoints/best_unet.pth').Path
$env:ATLDSD_EXPECTED_SHA256 = (Get-FileHash -Algorithm SHA256 $env:ATLDSD_WEIGHTS).Hash.ToLowerInvariant()
python -m uvicorn demo.app:app --host 127.0.0.1 --port 8000
```

如果权重存放在项目外，将第一行改成该文件的实际位置即可。浏览器打开 http://127.0.0.1:8000 。同一环境中也可运行 demo/start.cmd；如果使用项目内 .venv，它会优先使用该虚拟环境。按 Ctrl+C 停止前台服务。

API 为 GET /api/health 和 POST /api/predict（multipart/form-data 的 file 字段），交互文档位于 /docs。缺少权重或校验值时，页面仍可打开，并显示模型未就绪。

图片最大 8 MB、2,000 万像素，任意单边不超过 8,000 像素。后端返回叠加图、彩色掩膜、类别 ID 掩膜及面积统计。框架可能使用临时上传文件，该文件在读完后关闭；应用不保存图片和结果文件，前端下载由浏览器保存。

总病斑占比是四类病斑预测像素之和占预测叶片像素的比例；不是分类置信度。没有识别到叶片和没有检出病斑会分别显示。面积比例不直接转换成病情等级。

## 命令行预测

自行准备权重和可使用的图片后，执行：

```powershell
python predict.py --image './your_leaf.jpg' --weights './checkpoints/best_unet.pth'
```

命令行预测会把结果保存到 outputs/。这些运行产物属于用户本地输出，未包含在源码副本中；.gitignore 的文件允许清单会排除未审查的产物。命令行通过参数指定 checkpoint，不包含固定实验权重路径。


## 本地训练和验证

设置 ATLDSD_ROOT 指向你有权使用的数据集。目录结构为“类别目录/image/图片”和“类别目录/label/同名 PNG 标签”。标签保留类别索引 0–5。首次使用时，明确创建你自己的数据划分：

```powershell
$env:ATLDSD_ROOT = (Resolve-Path './data/ATLDSD').Path
python -m scripts.audit_dataset --regenerate-splits
python -m scripts.train_baseline --name my_run --dry-run
python -m scripts.train_baseline --name my_run
```

数据在项目外时，将环境变量改为实际位置。源码中包含可修改的通用默认参数（src/experiment.py），也可以用 --config 指向你自行准备的 JSON 配置。默认参数仅用于提供可运行入口，模型效果需自行验证。

dry-run 会检查现有划分和动态类别权重，不创建训练目录。每次正式训练使用一个新的 name；已有目录会拒绝覆盖。只有明确希望改变划分时才使用 --regenerate-splits，这会重新生成本地清单。

训练后，使用本次运行保存的实际参数快照进行验证：

```powershell
python -m scripts.evaluate_validation --config './runs/my_run/config_snapshot.json' --weights './runs/my_run/best_unet.pth' --output './outputs/my_validation'
```

验证入口读取本地划分中的验证集，拒绝缺失的清单或已占用的报告目录。训练参数快照、模型、划分、统计和报告均属于运行产物，未包含在源码副本中；文件允许清单不会将它们选入普通 Git 提交。

要把新训练的模型用于网页，设置 ATLDSD_WEIGHTS 和对应 SHA256，再启动网页服务。默认网页预处理输入为 256×256，使用六类 U-Net；训练时应保持这一结构及输入预处理一致。

## 训练日志与最终测试

需要同时显示并保存训练输出时，用下面的命令替代直接训练命令；同一个实验只启动一次：

```powershell
python -m scripts.run_training_logged --name my_logged_run
```

训练文件位于 runs/my_logged_run/，日志位于 outputs/training_logs/my_logged_run.log。日志入口不预先创建训练目录，避免与训练目录保护规则冲突。可以传入 --config、--epochs 等训练参数；同名日志或训练目录存在时会拒绝启动。前台按 Ctrl+C 会停止子进程。此入口负责记录输出，不提供断线后继续运行或断点恢复功能。

先用验证集选择模型并确定参数，再进行独立测试集评估：

```powershell
python -m scripts.evaluate_baseline --config './runs/my_logged_run/config_snapshot.json' --weights './runs/my_logged_run/best_unet.pth' --output './outputs/my_test'
```

测试入口使用与训练一致的缩放和归一化，读取固定测试划分，不重新生成清单，并拒绝覆盖已占用的报告目录。测试结果用于报告最终表现；继续根据测试结果调参会影响其作为独立评估的意义。日志、测试报告和图表仅在用户运行后生成，不包含在源码副本中。

## 方法及数据来源

### 原始数据：ATLDSD

本项目使用冯景泽、赵晓飞发布的《苹果树叶病分割数据集》（Apple Tree Leaf Disease Segmentation Dataset，ATLDSD），引用版本为 ScienceDB，2022，V1。

- [ScienceDB 数据记录页](https://www.scidb.cn/en/detail?dataSetId=0e1f57004db842f99668d82183afd578)
- DOI：[10.11922/sciencedb.01627](https://doi.org/10.11922/sciencedb.01627)
- CSTR：[31253.11.sciencedb.01627](https://cstr.cn/31253.11.sciencedb.01627)
- 数据许可：[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)，依据数据记录与此前核实的许可页面。

推荐引用：冯景泽，赵晓飞．苹果树叶病分割数据集[DS/OL]．V1．ScienceDB，2022．DOI：10.11922/sciencedb.01627．

原图和标签来自上述数据集，本项目不将它们声明为原创数据，也不提供数据下载镜像、标注或权重。使用者应到原记录页获取数据并遵守对应许可。

### 配置、日志和结果：本项目实验记录

公开配置是本项目自己的参数文件；公开训练历史与聚合指标由本项目运行程序产生，不是数据集官方基准成绩。历史日志只保留训练数值行，缺失快照及未核实副本明确标注。整理时从本地运行目录和训练归档读取文字材料，清理敏感路径，并保留实验数值。

每种文件的来源、处理方式、缺项及引用详见 [数据与实验文件来源](experiments/SOURCES.md)。

### 方法参考与外部尝试

U-Net 方法参考：Ronneberger, O.; Fischer, P.; Brox, T. U-Net: Convolutional Networks for Biomedical Image Segmentation. 2015. [原论文](https://arxiv.org/abs/1505.04597)。方法引用不替代实现代码的版权来源核查。

此前还尝试过 PlantSeg v3 的外部混合适配，未获得满足项目评估约束的候选。来源与版本许可见 [来源说明](experiments/SOURCES.md#4-外部数据尝试plantseg)；仅保留文字概述，不提供该尝试的实验附件。

数据集的许可声明不作为本仓库源码的统一许可证；源码许可将在来源与版权归属核实后确定。

## 历史实验材料

经过整理的历史配置、训练曲线、训练日志摘录及聚合评估结果位于 [experiments/README.md](experiments/README.md)。它们保留原实验数值，并标明缺项；不包含样本清单、图片、权重或完整服务器日志。

## 上传 GitHub 与学习 Git

Git 上传指南、命令速查与动手练习作为本地学习资料单独保存，不纳入本仓库。

辅助脚本位于 `scripts/check_publication.py`、`scripts/update_record_manifest.py` 和 `scripts/create_git_learning_lab.py`。前两者帮助检查发布副本与维护实验文件哈希；学习脚本在新建临时目录练习，不连接远程。检查范围与限制见脚本说明，技术检查不代替许可与版权核实。
