# InsertAnything：分侧光学校准、重新采集与角度预测 v2

> 2026-10-12：按 TCP 中心、多个接近方向和平面偏移生成独立接触姿态的新流程见 [v3 手册](README_insertanything_tactile_tcp_v3.md)。本页继续记录旧的物理闭爪采集与 panda_hand 旋转模型，不与 v3 TCP 标签混用。

更新：2026-10-12。代码、数据和模型仍在 `/data/nju/Lincao/InsertAnything_tactile`；原 InsertAnything / UniVTAC 的配置、代码及已有实验保留。本文是当前校准与姿态训练入口；[v1 结构、字段及历史结果](README_insertanything_tactile_pose.md)继续保留。

## 已接入的分侧校准

默认 `cfg.tactile.calibration_mode="geometry_per_side"`。每次构建场景，从同一 USD 坐标系计算胶垫前表面到各自相机像平面的距离：左 **28.246235 mm**、右 **27.950882 mm**。相机裁剪面保持 24–34 mm，与光学零点分别配置。

```text
height_map_mm = clip(camera_depth_mm - zero_side_mm, -4, 0)
press_depth_mm = -height_map_mm
```

无效或超出裁剪面的深度不产生接触。校准高度图先进入 TacEx 的压入量计算和 Taxim，再生成 RGB；保存的深度、图像和接触状态使用同一基准。没有复制或翻转另一侧图像来生成数据，也没有改变胶垫的物理位置、质量或原任务动作接口。

校准实现位于 [tactile_calibration.py](/data/nju/Lincao/InsertAnything_tactile/source/tactile_calibration.py) 和 [env.py](/data/nju/Lincao/InsertAnything_tactile/source/insertanything_tactile/env.py)。几何变换统一取 authored USD，避免将 Fabric 运行时位姿与缓存的相机 USD 位姿混合。

这是当前 **PhysX 刚体胶垫的分侧几何光学零点校准**，不是接触力标定、软胶有限元标定或真实传感器的逐像素标定。

### 验收结果

方孔 II、seed 0、8 步实际运行；再移走物体，以相同指位检查空载。

| 指标 | 左 | 右 |
|---|---:|---:|
| 同一帧旧 28 mm 基准最大压入 / mm | 0.000 | 0.317 |
| 新运行时最大压入 / mm | 0.238 | 0.268 |
| 压入大于 0.02 mm 的像素 | 16,935 | 17,063 |
| 空载最大压入 / mm | 0 | 0 |
| 空载接触像素 | 0 | 0 |
| 运行时 RGB 与独立重渲染的最大像素误差 | 0 | 0 |

镜像对齐后的左右接触区域 IoU **0.97686**。[报告](/data/nju/Lincao/InsertAnything_tactile/outputs/calibration_v2_validation_20261012/report.json)。

![分侧零点、实际运行和空载检查](/data/nju/Lincao/InsertAnything_tactile/outputs/calibration_v2_validation_20261012/calibration_validation.png)

重新执行短检查，输出目录须不存在：

```bash
cd /data/nju/Lincao/InsertAnything_tactile
./scripts/python.sh scripts/validate_tactile_calibration.py --headless --output outputs/calibration_v2_check_new
```

## 数据与版本

新格式为 `tactile_pose_v2`，在 v1 的 38 个同步字段上新增两侧 `tactile/{left,right}/camera_depth_mm`，共 **40 个字段**。相机深度是像平面距离，单位 mm；非有限无命中值以远裁剪面 34 mm 保存。RGB、实际手指位置、实际物体/夹爪位姿、相对旋转真值、步号、时间、环境/抓取/物体编号、种子及接触/状态标志均保留。

HDF5 `metadata_json`、同目录 `metadata.json`、划分文件和 checkpoint 都记录 `tactile_calibration`，包含模式、左右零点、裁剪面、胶垫范围、图像尺寸和校准 ID。本次 ID：

```text
1056c7a9a5ce18318c2c29f193e75f2d0ddabd4779747f583e376f3dfcb5213f
```

训练拒绝混用不同校准 ID 的文件；HDF5 推理拒绝将 v1 权重用于 v2 图像。独立 PNG 不携带此元数据，调用者应保证两图与权重采用同一校准、顺序及开度定义。v1 文件和权重保持可读；`legacy_shared` 是复现历史 28 mm 光学模式的显式选项。

`collect_pose_data.py`、`capture.py`、`record_eval.py`、`render_scene.py` 支持 `--tactile_calibration geometry_per_side`（默认）或 `--tactile_calibration legacy_shared`。原插入策略仍只用原观测，其 checkpoint 与此处的角度估计 checkpoint 用途不同。

有效条件仍为双侧各至少 64 个像素压入大于 0.02 mm，状态有限、物体距夹爪原点小于 0.25 m、实际手指关节位移之和在 `(0,0.081)` m。所有无效样本照样保存；评估精度只针对有效双侧接触，另报告有效率。

## 本轮采集与训练

已完成以下实验，旧数据和 checkpoint 均保留：

- 主采集：600 次抓取、4,800 组同步双侧图像；562 次抓取中的 4,461 组有效（92.94%）。[数据与检查](/data/nju/Lincao/InsertAnything_tactile/outputs/pose_lhole_v2_600grasps_20261012/validation.json)。
- 独立抓取划分：训练 394 次 / 3,117 帧，验证 84 次 / 672 帧，测试 84 次 / 672 帧。
- 新种子复测：120 次抓取、960 组图像；108 次抓取中的 855 组有效（89.06%）。[数据与检查](/data/nju/Lincao/InsertAnything_tactile/outputs/pose_lhole_v2_fresh120_20261012/validation.json)。
- 训练 60 轮，按验证误差选中第 50 轮；[当前 best.pt](/data/nju/Lincao/InsertAnything_tactile/outputs/pose_lhole_v2_train_20261012/best.pt)。

主采集 40 个字段的同步、有限值、校准深度和真值检查通过，旋转矩阵与 SciPy 独立重算的最大元素误差为 `7.87e-7`。实际相对平均旋转的旋转向量覆盖：X [-11.82,17.28]°、Y [-32.06,50.62]°、Z [-41.36,45.28]°；不等于欧拉角覆盖或完整三维姿态均匀采样。

以下均为有效接触帧的 SO(3) 三维旋转误差。历史 v1 是不同规模、不同种子的参考，不能作为单变量校准消融。

| 评估 | 独立抓取 / 帧 | 平均 | 中位数 | P95 | ≤5° | ≤10° |
|---|---:|---:|---:|---:|---:|---:|
| 历史 v1 测试 | 17 / 136 | 12.12° | 7.18° | 34.54° | 23.53% | 64.71% |
| v2 保留测试 | 84 / 672 | 6.16° | 2.73° | 25.50° | 70.83% | 82.29% |
| v2 全新抓取复测 | 108 / 855 | 7.22° | 2.83° | 40.98° | 70.18% | 85.85% |

复测常数基线（始终输出训练集平均旋转）的平均误差为 **19.13°**，模型为 **7.22°**，说明模型能利用输入估计姿态。但模型的复测 P95 **40.98°** 仍高于该基线的 **38.85°**；9/108 次抓取的平均误差超过 20°，其中 6 次超过 40°，最差帧为 **73.43°**。这些大误差样本全部保留。

训练平均误差 0.16°，验证 5.69°，保留测试 6.16°，新种子复测 7.22°，仍有明显泛化差距。按抓取等权平均，复测误差为 **7.39°**，抓取组 bootstrap 95% 区间为 **[5.13,10.09]°**。残差旋转向量 XYZ 平均绝对分量分别为 **1.43°、2.61°、5.06°**；Z 分量的误差最大。

**本轮结论：多数有效抓取已能获得几度级姿态估计，但大误差尚未解决，还不能作为稳定的精密插入角度反馈。** 可见的一些失败样例只呈现局部接触边缘；这提示需要进一步分析可观测性并补充此类抓取，当前证据不能将所有错误归结为单一原因。

[训练指标](/data/nju/Lincao/InsertAnything_tactile/outputs/pose_lhole_v2_train_20261012/metrics.json) · [保留集 CPU 复核](/data/nju/Lincao/InsertAnything_tactile/outputs/pose_lhole_v2_test_20261012/metrics.json) · [新种子复测指标](/data/nju/Lincao/InsertAnything_tactile/outputs/pose_lhole_v2_fresh_test_20261012/metrics.json)

![全新抓取的角度真值与预测](/data/nju/Lincao/InsertAnything_tactile/outputs/pose_lhole_v2_fresh_test_20261012/angle_predictions.png)

下图每列为同一帧，**上行为左触觉、下行为右触觉**，从误差最低到最高的抓取分位数选样，最右列为最差抓取。

![新校准的左右触觉与预测误差](/data/nju/Lincao/InsertAnything_tactile/outputs/pose_lhole_v2_fresh_test_20261012/tactile_prediction_examples.png)


网络保持 v1：左右共享 CNN → 按左右顺序拼接 → 实测 `q_left+q_right` → MLP → 6D 旋转表示 → SO(3)。参数仍为 **461,574**，不将真值、随机种子、环境或抓取编号输入模型。

[pose_v2.json](/data/nju/Lincao/InsertAnything_tactile/configs/pose_v2.json)：60 epochs，batch 64，AdamW，初始学习率 0.0005，余弦下降至 0.00002，weight decay 0.0001，训练种子 0。预处理后的图像在内存缓存，已检查与非缓存加载逐值一致；不会改变输入图像或划分。

按有效独立抓取约 70%/15%/15% 分训练/验证/测试；同一次抓取的所有帧归同一集合。`best.pt` 只按验证集平均 SO(3) 测地角选择。额外复测的 UUID 和随机种子均与训练采集不重叠，且不参与选模。

请求的初始化范围是 XYZ ±20°/±20°/±35°；实际物理闭爪会使朝向回正，覆盖范围以 `validation.json` 和 `rotation_coverage.png` 为准。相对旋转标签仍由实际刚体位姿计算，不能用请求扰动角代替。

## 重新采集、训练、评估命令

以下为可复用的新输出目录示例。每行是一条完整命令，已有输出目录会被拒绝。复用已配置的 Isaac Sim 5.1 / Isaac Lab 2.3 / Python 3.11；采集需 GPU 和相机渲染，离线训练/评估无需启动 Isaac Sim。

```bash
cd /data/nju/Lincao/InsertAnything_tactile
./scripts/python.sh scripts/collect_pose_data.py --headless --num_envs 4 --grasps 600 --frames_per_grasp 8 --seed 20000 --output outputs/pose_lhole_v2_new
./scripts/python.sh scripts/inspect_pose_data.py outputs/pose_lhole_v2_new/samples.h5
./scripts/python.sh scripts/train_pose.py --data outputs/pose_lhole_v2_new/samples.h5 --config configs/pose_v2.json --device cuda:0 --output outputs/pose_lhole_v2_train_new
./scripts/python.sh scripts/evaluate_pose.py --checkpoint outputs/pose_lhole_v2_train_new/best.pt --split test --output outputs/pose_lhole_v2_test_new
./scripts/python.sh scripts/report_pose_results.py --checkpoint outputs/pose_lhole_v2_train_new/best.pt --evaluation outputs/pose_lhole_v2_test_new
```

用另一组种子做新的独立抓取复测：

```bash
./scripts/python.sh scripts/collect_pose_data.py --headless --num_envs 4 --grasps 120 --frames_per_grasp 8 --seed 50000 --output outputs/pose_lhole_v2_fresh_new
./scripts/python.sh scripts/inspect_pose_data.py outputs/pose_lhole_v2_fresh_new/samples.h5
./scripts/python.sh scripts/evaluate_pose.py --checkpoint outputs/pose_lhole_v2_train_new/best.pt --data outputs/pose_lhole_v2_fresh_new/samples.h5 --output outputs/pose_lhole_v2_fresh_test_new
./scripts/python.sh scripts/report_pose_results.py --checkpoint outputs/pose_lhole_v2_train_new/best.pt --evaluation outputs/pose_lhole_v2_fresh_test_new
```

使用本轮已经训练好的权重推理：

```bash
./scripts/python.sh scripts/predict_pose.py --checkpoint outputs/pose_lhole_v2_train_20261012/best.pt --data outputs/pose_lhole_v2_fresh120_20261012/samples.h5 --row 0
```

已经验证上述行的 HDF5 与独立 PNG＋实际开度入口，预测矩阵逐值一致；旧 v1 权重读取 v2 数据会明确拒绝校准不匹配。

推理只读取左右 RGB 和实测开度，输出 `R_GO`、wxyz 四元数及显示用 XYZ 欧拉角。误差指标采用三维 SO(3) 测地角，避免欧拉角 ±180° 跳变；`residual_rotvec_mae_xyz_deg` 是夹爪坐标系内残差旋转向量三轴的平均绝对分量。绘图中三轴角度是相对固定训练集平均旋转的旋转向量分量，并非欧拉角。

## 文件与检查

- 校准验证：`outputs/calibration_v2_validation_20261012/`，含报告、原始数组、旧/新/空载图。
- 采集：`samples.h5`、`metadata.json`；检查生成 `validation.json`、`rotation_coverage.png`、`grasp_preview.png`。
- 训练：`best.pt`、`last.pt`、`config.json`、`splits.json`、`history.jsonl`、`metrics.json`、`test_predictions.npz`。
- 独立评估：`metrics.json`、`predictions.npz`、`predictions.csv`；CSV 对应源文件、行号、抓取组及逐帧误差。
- 结果图：`angle_predictions.png`、`tactile_prediction_examples.png`、`analysis.json`。样例按抓取误差分位数从最好到最差选取，包含最差样例；置信区间按抓取组自助采样，而非把连续帧当成独立样本。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_pose.py tests/test_calibration.py
./scripts/python.sh scripts/verify_isolation.py
```

当前边界：L 形插销、仿真中可稳定夹持的有限朝向范围，不能外推至完整 SO(3)、其他物体或真实传感器。校准修复了输入的系统偏差；本轮同时扩充数据并调整优化配置，因此与历史 v1 指标的差异不能全部归因于校准本身。

本次回归：11 项离线检查通过，原 InsertAnything 48 个受保护文件与 UniVTAC 1,886 个受保护文件共 1,934 个均未变化。
