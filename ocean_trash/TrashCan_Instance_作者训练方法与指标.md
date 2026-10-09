# TrashCan-Instance 作者训练方法与指标整理

整理日期：2026-10-09。仅讨论当前使用的 `instance_version`，不混入 Material 版本或其他论文的实验结果。

## 1. 资料出处与阅读范围

作者：Jungseok Hong、Michael Fulton、Junaed Sattar，明尼苏达大学。

- [作者实验室项目主页](https://irvlab.cs.umn.edu/projects/tackling-marine-debris)：确认项目、论文与数据集之间的关系。
- [原始论文 PDF](https://arxiv.org/pdf/2007.08097)：全文共 6 页；重点核对第 3 节训练方法、表 1 总体结果及图 3 分类结果。
- [论文 HTML](https://arxiv.org/html/2007.08097)：便于直接定位训练段落和指标表。
- [你提供的官方数据集页面](https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7)。本次访问返回 HTTP 403，未能读取其下载附件。
- [明尼苏达大学数据集登记页](https://experts.umn.edu/en/datasets/trashcan-10-an-instance-segmentation-labled-dataset-of-trash-obse/)：确认作者还发布了 `instance_checkpoints.zip`，内含 Instance 实验的网络配置与权重；本次未取得该压缩包。
- 本地 `data/trashcan_instance_raw/README.txt`、两份原始 COCO JSON 和转换后的 YOLO 标签：用于核对当前数据的数量、类别与划分。

下文区分“论文明确给出”“本地统计”和“我们项目的理解”。未读取到的训练配置不当作已确认事实。

## 2. 作者如何训练

作者用 PyTorch 的 Detectron2 实现两组基线。两者均以 COCO 预训练权重初始化，在 TrashCan 数据上继续训练，而非从随机权重开始。

| 项目 | 检测基线 | 实例分割基线 |
| --- | --- | --- |
| 模型 | Faster R-CNN | Mask R-CNN |
| 主干 | ResNeXt-101-FPN（X-101-FPN） | X-101-FPN |
| 预训练数据 | COCO | COCO |
| 训练步数 | 10,000 iterations | 15,000 iterations |
| GPU | NVIDIA Titan XP | NVIDIA Titan V |
| 评估方式 | COCO 检测指标 | COCO 实例分割指标 |

以上来自[原始论文第 3 节](https://arxiv.org/html/2007.08097#S3)。作者将结果定位为初始基线，不是该数据集的最佳成绩。

按任务理解其流程：读取 Instance 的图像和 COCO 标注，用预训练模型微调，再按对应任务评价输出。Faster R-CNN 学习目标类别和边界框；Mask R-CNN 还学习每个目标的像素掩膜。你目前的 YOLO 多边形标签对应后者的实例分割任务。

**iterations 是参数更新步数，不是 epochs。** 一次 iteration 通常处理一批图片；epoch 表示遍历一次训练集。近似关系为：

```text
等效 epochs ≈ iterations × 总 batch size / 训练图片数
```

因为作者 batch size 尚未确认，不能把 15,000 iterations 写成确定的 epoch 数，更不能直接给 YOLO 设置 `epochs=15000`。

### 尚未确认的训练细节

论文没有列出学习率、优化器、batch size、输入尺寸、学习率衰减、warmup、增强配置、冻结策略、随机种子、Detectron2 版本及最终 checkpoint 的选择规则。不得拿今天的框架默认值补成作者参数。

若以后要精确复现实验，需进一步读取官方 `instance_checkpoints.zip` 中两种模型的配置与训练记录。[大学登记页](https://experts.umn.edu/en/datasets/trashcan-10-an-instance-segmentation-labled-dataset-of-trash-obse/)明确说明该包包含配置和权重，但不保证包含完整训练脚本或日志。

## 3. 作者最终指标：仅 Instance 版本

下表逐项核对自[论文表 1](https://arxiv.org/html/2007.08097#S3)，数值采用论文的 0–100 标度。

| 模型及评估任务 | AP | AP50 | AP75 | AP 小目标 | AP 中目标 | AP 大目标 |
| --- | --- | --- | --- | --- | --- | --- |
| Mask R-CNN：实例分割 | **30.0** | **55.3** | 29.4 | 23.2 | 31.7 | 48.6 |
| Faster R-CNN：目标检测 | **34.5** | **55.4** | 38.1 | 27.6 | 36.2 | 51.4 |

这两行分别对应掩膜任务和边界框任务。不能用 34.5 与 30.0 的差值判断“检测模型比分割模型准确多少”，因为评价对象不同。

### 这些数字代表什么

依据 [COCO 官方评估代码](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)：

- **AP**：在 IoU=0.50、0.55、…、0.95 的 10 个阈值上平均，与常说的 mAP50–95 对应。
- **AP50 / AP75**：分别只在 IoU=0.50 / 0.75 下评价。
- **IoU**：预测区域与真实区域的交集面积除以并集面积。检测用框，分割用掩膜。
- **小、中、大目标 AP**：按标注目标面积分组；标准 COCO 使用 32²、96² 像素面积作为分组边界，不是按实际物体的厘米尺寸。

例如论文 AP=30.0，对应常见 0–1 标度约为 0.300。它不是“30% 的图片识别正确”，也不是分类 accuracy。

论文图 3 提供每类 AP 的柱状图，但没有列出精确数值表。本文不把目测柱高伪装成精确类别成绩。论文也没有报告可用于本项目的 FPS、推理延迟或三维定位误差。

## 4. 与你当前本地数据的对应关系

以下为读取本地 JSON 得到的统计，不是从 README 表格抄录；README 个别总数与 JSON 不一致时，以实际训练输入为准。

| 项目 | 训练集 | 验证集 | 总计 |
| --- | --- | --- | --- |
| 图片 | 6065 | 1147 | 7212 |
| 原始 COCO 实例标注 | 9540 | 2588 | 12128 |
| 当前 YOLO 转换后的实例行 | 9540 | 2587 | 12127 |

转换器去掉了验证集 1 条相同边界框的重复记录。因此当前 YOLO 标签与原始 COCO 标注不是逐条完全一致；严格对照论文时，应以原始 COCO 标注作为评价真值。

当前数据有 **22 类：14 类垃圾、6 类动物、植物、ROV**。总体 AP 包含全部类别，不能称为“仅垃圾的 AP”。

| YOLO 编号 | 原始类别 | 中文理解 | 训练实例 | 验证实例 |
| --- | --- | --- | --- | --- |
| 0 | rov | 水下机器人／人为放置的装备 | 2633 | 684 |
| 1 | plant | 植物 | 405 | 102 |
| 2 | animal_fish | 鱼类 | 611 | 153 |
| 3 | animal_starfish | 海星 | 262 | 136 |
| 4 | animal_shells | 贝类 | 188 | 61 |
| 5 | animal_crab | 螃蟹 | 246 | 63 |
| 6 | animal_eel | 鳗鱼 | 259 | 84 |
| 7 | animal_etc | 其他动物 | 170 | 65 |
| 8 | trash_clothing | 衣物垃圾 | 65 | 17 |
| 9 | trash_pipe | 管道垃圾 | 114 | 42 |
| 10 | trash_bottle | 瓶子 | 100 | 26 |
| 11 | trash_bag | 袋子 | 727 | 181 |
| 12 | trash_snack_wrapper | 零食包装 | 67 | 17 |
| 13 | trash_can | 罐子 | 366 | 93 |
| 14 | trash_cup | 杯子 | 47 | 12 |
| 15 | trash_container | 容器 | 407 | 103 |
| 16 | trash_unknown_instance | 未知类型垃圾 | 2203 | 553 |
| 17 | trash_branch | 树枝 | 268 | 68 |
| 18 | trash_wreckage | 残骸 | 130 | 35 |
| 19 | trash_tarp | 防水布 | 90 | 31 |
| 20 | trash_rope | 绳索 | 88 | 29 |
| 21 | trash_net | 渔网 | 94 | 33 |

这里的“Instance 版本”表示垃圾按物品类型区分类别，例如袋子、瓶子；不表示另一版本没有分割标签。[作者项目主页](https://irvlab.cs.umn.edu/projects/tackling-marine-debris)说明两版本的差别在类别配置。

### 数据划分需要注意

6065 / 7212 ≈ 84.10%，1147 / 7212 ≈ 15.90%。这是本地文件的实际图片比例；README 中许多类别约 80/20 的数字指实例分布，不能直接当作图片划分比例。

按 JSON 文件名中 `_frame` 前的视频编号统计：训练集 310 个编号、验证集 115 个编号，其中 **113 个编号重叠**。这是文件名层面的本地证据，不能证明帧内容重复，但足以说明当前划分不是视频编号完全隔离的划分。

与论文基线比较，应保留官方 train/val。若要验证对全新视频的泛化能力，可以另外建立视频互斥的测试集并单独报告成绩，不把新划分结果冒称为论文同条件比较。

## 5. 我们训练时如何对照

以下是本项目的实验建议，**不是作者已使用的参数**：

1. 先保留全部 22 类和官方划分训练 YOLO 实例分割，形成可与作者任务对应的基线。
2. 同时记录 Box 与 Mask 的 mAP50、mAP50–95；分割主要参考作者 Mask AP=30.0、AP50=55.3，检测主要参考 Faster R-CNN 的 Box AP=34.5、AP50=55.4。
3. 若要严谨声明超过作者，导出预测并以原始验证集 COCO JSON 做相同任务的 COCOeval。YOLO 内置 mAP 与 COCO AP 名称对应，但默认评估实现、掩膜处理及预测截断设置不一定完全相同。
4. 保留每类指标，额外计算垃圾类别的表现。ROV 和未知垃圾实例较多，杯子、衣物等类别样本少；总体成绩不能替代少样本垃圾类别的核查。
5. 单独记录模型、预训练权重、epochs、batch、输入尺寸、优化器、学习率、增强、随机种子、GPU 与版本，方便复现自己的训练。

作者的实验解决图像中的类别、框和掩膜预测，没有提供从这批图像训练三维定位的方法。当前数据集训练出的模型可以提供后续定位所需的目标区域；距离和三维坐标仍需另外的测量信息。

## 6. 当前项目文件

- 原始数据：`data/trashcan_instance_raw/`
- YOLO 数据配置：`data/trashcan_instance_yolo/trashcan_instance.yaml`
- 标签可视化：`data/trashcan_instance_review/train/`、`data/trashcan_instance_review/val/`
- 转换代码：`src/dataprocess/convert_trashcan_to_yolo.py`
- 可视化代码：`src/dataprocess/visualize_yolo_segments.py`

本次只整理作者实验及与你本地数据的对应关系，没有启动模型训练。
