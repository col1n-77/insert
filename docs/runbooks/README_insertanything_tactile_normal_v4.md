# InsertAnything v4：竖直插销、夹持法向在手偏角、单角度预测

更新：2026-10-12。当前采集、训练和预测入口；代码位于 `/data/nju/Lincao/InsertAnything_tactile`。原任务、旧数据和旧模型保留。

## 1. 本次明确采用的约束

用户确认的方式是：**插销轴线始终竖直，夹爪绕夹持法向倾转，网络只预测一个在手偏角。**

- 插销局部 +Z 始终与世界 +Z 同向，固定世界 yaw=180°；物体世界姿态不随样本随机旋转。
- 夹持法向是两指连线，即真实 TCP 的 +Y/−Y 轴。通过实际 Panda 七关节的逆运动学改变夹爪姿态，机器人基座保持不动。
- TCP 位置固定在当前任务初始化位置附近，夹爪只绕法向倾转；不再采多个接近方向或绕接近轴作全周旋转。
- 在手偏角 θ 范围为 **−15°～+15°**。这是物体相对 TCP、绕 +TCP Y 的右手有符号角度；不是世界 yaw。
- 保留 TCP XY 平面内 ±2 mm 的物体中心平移、0.2～1 mm 的压入目标提议。平移/压入是干扰变量，模型只回归 θ。
- 每个姿态独立采样，只保存一对同步触觉图像。压入量和位移仍按关节限位、双侧有效接触筛选，接受后的深度/平移分布并非均匀。

真实 TCP 为 `panda_fingertip_centered`。物体根原点与包围盒几何中心分别保存，几何中心默认放在 TCP Z=0、XY±2 mm 附近。名义插入姿态的 TCP +Z 向下、物体 +Z 向上，因此零偏角的物体相对 TCP 旋转矩阵不是单位矩阵。

```text
R_TO = R_WT.T @ R_WO
p_TO = R_WT.T @ (p_WO - p_WT)
R_TO(0) = RotY(180°)
R_TO(θ) = RotY(θ) @ R_TO(0)
R_WO = RotZ(180°)              # 固定世界姿态，物体 +Z 始终向上
R_WT(θ) = R_WO @ R_TO(θ).T     # 通过实际手臂关节 IK 实现
```

四元数仍为 wxyz，位置单位 m。标签由写入并渲染后的实际世界位姿计算，不使用采样请求值充当真值。每帧额外检查物体竖直性、非目标轴旋转、实际角度与请求角度的差，以及基座是否移动。

采集是**静态几何接触渲染**：直接写入实际关节/物体状态，再执行 forward 和渲染，样本之间不积分物理。不将刚体几何重叠解释成真实软胶形变、接触力或稳定抓取验证。采集实例中孔件暂移离近场；原插入任务配置和策略未改变。

## 2. 已生成的数据

| 用途 | 路径 | 有效样本 |
| --- | --- | ---: |
| 训练母集，root seed=80000 | `/data/nju/Lincao/InsertAnything_tactile/outputs/normal_angle_v4_train600_upright_20261012` | 600，每物体 100 |
| 独立复测，root seed=90000 | `/data/nju/Lincao/InsertAnything_tactile/outputs/normal_angle_v4_fresh120_20261012` | 120，每物体 20 |
| 新模型 | `/data/nju/Lincao/InsertAnything_tactile/outputs/normal_angle_v4_model_20261012/best.pt` | 1 个角度输出 |
| 独立复测结果 | `/data/nju/Lincao/InsertAnything_tactile/outputs/normal_angle_v4_fresh_eval_20261012` | 120 条逐样本预测 |

六个子目录：`CircleHole-I`、`SquareHole-II`、`LHole-III`、`TriangleHole-IV`、`HexagonHole-III`、`HexagonHole-IV`。每个子目录包含 `samples.h5`、`metadata.json`、`proposals.jsonl`、角度区间的触觉 PNG/场景预览。

每个物体均匀分配 5 个角度区间：`[-15,-9]、[-9,-3]、[-3,3]、[3,9]、[9,15]`。训练母集每区间 20 个有效姿态，独立复测每区间 4 个。全部 **720 对触觉 RGB，即 1,440 张触觉图像**存于 HDF5；另有每区间的 PNG 示例。

圆插销在这个固定夹持方位下，因两胶垫闭合后的最小间隙约 7.144 mm，深度接受范围主要为 0.2～0.4 mm；三角插销的训练母集接受到约 0.657 mm。L/方/六边形均有接近 1 mm 的有效样本。超过当前几何可达范围的提议有明确拒绝记录，不越过关节限位。

两批数据均通过 `inspection/validation.json` 的验收。600 组母集的物体轴线与世界竖直方向误差在浮点零附近，非目标轴最大残差约 **0.000090°**，机器人基座变化为 0。这些是位姿约束验收误差，不是模型预测误差。

[新独立批次六物体双侧图像](/data/nju/Lincao/InsertAnything_tactile/outputs/normal_angle_v4_fresh120_20261012/inspection/stereo_overview.png) · [L 插销五档偏角：场景＋左右触觉](/data/nju/Lincao/InsertAnything_tactile/outputs/normal_angle_v4_fresh120_20261012/inspection/LHole-III_angles.png)

训练母集的首个可选场景全景曾遇到视口预热黑帧，检查图会明确显示场景预览不可用；其触觉相机和姿态数据验收正常。场景预热已修复，独立复测批次的六个首帧场景均已确认非空；推荐查看上方新批次图集。

## 3. 只预测一个角度的网络与结果

```text
左 RGB ─共享 CNN─128 维特征─┐
                           ├─拼接＋实际夹爪开度─MLP─1 个 θ（度）
右 RGB ─共享 CNN─128 维特征─┘
```

CNN 四层通道为 24/48/96/128，输入缩放到 120×160，最后池化并编码为 128 维；两侧按固定顺序拼接实际开度，MLP 为 257→256→128→1，`tanh×15` 输出一个有符号角度。共 **460,929 参数**。六种已知物体共同训练一个模型，物体 ID 不作为网络输入。

输入仅为 `tactile/left/rgb`、`tactile/right/rgb`、`gripper/opening_joint_sum_m`。世界位姿、采样角、机械臂关节和物体 ID 不参与前向预测。完整位姿保留用于验收，监督目标仅为 `target/in_hand_angle_deg`。

按独立姿态、物体和角度区间分层划分：**420 训练 / 90 验证 / 90 留出测试**。80 个 epoch，AdamW 初始学习率 0.0005，余弦降至 0.00002，batch 64，归一化角度 MSE。仅按验证集选择最佳 epoch **75**，留出测试在模型选定后计算。

| 数据集 | 样本数 | MAE | 中位数误差 | P95误差 | 误差≤2° |
| --- | ---: | ---: | ---: | ---: | ---: |
| 训练 | 420 | 0.347° | 0.200° | 1.265° | 99.05% |
| 验证 | 90 | 0.889° | 0.651° | 2.182° | 90.00% |
| 留出测试 | 90 | 1.021° | 0.860° | 2.571° | 87.78% |
| 独立新种子测试 | 120 | 0.751° | 0.579° | 1.997° | 95.00% |

独立测试按物体统计（各 20 组）：

| 物体 | MAE | P95误差 |
| --- | ---: | ---: |
| CircleHole-I | 0.718° | 1.419° |
| SquareHole-II | 0.836° | 2.028° |
| LHole-III | 0.944° | 2.849° |
| TriangleHole-IV | 0.910° | 2.108° |
| HexagonHole-III | 0.575° | 1.558° |
| HexagonHole-IV | 0.520° | 1.623° |

独立 120 组与训练母集的数据 UUID 和所有已渲染采样种子均无重叠。常数“训练角度均值”基线在该批上的 MAE 为 **7.468°**。这些结果覆盖当前六种已知插销、±15° 单轴偏角、当前静态仿真与校准，不是未知物体或实物部署结果。

[预测角度与真值散点图](/data/nju/Lincao/InsertAnything_tactile/outputs/normal_angle_v4_fresh_eval_20261012/angle_predictions.png) · [从最好到最差误差的触觉预测示例](/data/nju/Lincao/InsertAnything_tactile/outputs/normal_angle_v4_fresh_eval_20261012/tactile_prediction_examples.png) · [逐样本 CSV](/data/nju/Lincao/InsertAnything_tactile/outputs/normal_angle_v4_fresh_eval_20261012/predictions.csv)

## 4. 采集、训练、评估和预测指令

前提是现有 Isaac Sim 5.1 / Isaac Lab 2.3 / Python 3.11 环境和 RTX GPU 可用。沿用 `scripts/python.sh`，无需安装或修改环境。

重新采集，自动新建带时间戳目录：

```bash
cd /data/nju/Lincao/InsertAnything_tactile
./scripts/python.sh scripts/collect_normal_angle_data.py \
  --task all --samples_per_task 100 \
  --angle_range_deg -15 15 --angle_bins 5 \
  --offset_xy_mm 2 2 --depth_range_mm 0.2 1.0 \
  --seed 81000 --headless
```

`--task LHole-III` 可单采 L 形。`--samples_per_task` 为有效姿态数，必须是 `--angle_bins` 的整数倍。每名额默认最多 1,000 次提议，无法完成则保存失败状态并退出。可指定新 `--output`；已有目录会拒绝覆盖。其他偏角范围虽可配置（限 ±45° 内），本次只验收了 ±15°，扩大范围需要重新做接触与 IK 验收。

使用本次 600 组训练母集重训（输出目录须未存在）：

```bash
./scripts/python.sh scripts/train_normal_angle.py \
  --data outputs/normal_angle_v4_train600_upright_20261012 \
  --epochs 80 --batch_size 64 --seed 0 \
  --output outputs/normal_angle_v4_model_new
```

在独立数据上评估已有模型：

```bash
./scripts/python.sh scripts/evaluate_normal_angle.py \
  --checkpoint outputs/normal_angle_v4_model_20261012/best.pt \
  --data outputs/normal_angle_v4_fresh120_20261012 \
  --output outputs/normal_angle_v4_eval_new --device cpu
```

单样本只预测 θ（附加 `--show_label` 才同时显示真值和误差）：

```bash
./scripts/python.sh scripts/predict_normal_angle.py \
  --checkpoint outputs/normal_angle_v4_model_20261012/best.pt \
  --h5 outputs/normal_angle_v4_fresh120_20261012/LHole-III/samples.h5 \
  --row 0 --show_label
```

该实测样本：预测 **−9.5565°**，真值 **−10.1352°**，绝对误差 **0.5787°**。单帧推理与独立批量评估的预测一致。该脚本读取已保存观测；没有在本轮接入策略闭环控制。

离线核验并查看数据（输出目录须为新目录）：

```bash
./scripts/python.sh scripts/inspect_normal_angle_data.py \
  outputs/normal_angle_v4_fresh120_20261012 \
  --output /tmp/normal_angle_v4_review
```

## 5. 关键字段、兼容性与代码位置

Schema：`tactile_tcp_normal_v4`。训练和推理会检查 schema、光学校准及角度定义；v1/v2/v3 数据或旧三维旋转权重不能混入。

| 字段 | 含义 |
| --- | --- |
| `target/in_hand_angle_deg` | **唯一预测目标**，绕 +TCP Y 的实际有符号偏角 |
| `target/position_to_m`、`target/rotation_to`、`target/quaternion_to_wxyz` | 仍保留的实际完整相对位姿，用于检查 |
| `object/position_w_m`、`object/quaternion_wxyz`、`object/axis_w` | 实际物体世界位姿及轴线 |
| `tcp/position_w_m`、`tcp/quaternion_wxyz` | 实际 TCP 世界位姿 |
| `tactile/{left,right}/rgb` | 原生 320×240 RGB，uint8，N×H×W×3 |
| `tactile/{left,right}/press_depth_mm`、`camera_depth_mm` | 几何压入图、相机平面深度 |
| `gripper/finger_joint_position_m`、`opening_joint_sum_m` | 两指实际关节位置、模型使用的实际关节位移和 |
| `gripper/gel_front_gap_m` | 含安装偏移的胶面间距，区别于关节位移和 |
| `sampling/requested_angle_deg`、`rotation_axis_t` | 请求角度及法向，仅作采样记录 |
| `sampling/ik_*` | IK 次数、写入后实际 TCP 位置和旋转误差 |
| `quality/object_vertical_error_deg` | 物体轴线相对世界竖直的误差 |
| `quality/off_axis_rotation_deg` | 单法向旋转之外的旋转残差 |
| `quality/robot_base_max_change` | 基座位姿矩阵相对初值最大元素变化 |
| `quality/sample_valid`、`status_code` | 有效性与原因标志；新增 128 位表示竖直/单轴约束无效 |
| `ids/*`、`random/seed`、`time/*` | 环境、物体、姿态、角度区间、种子、真实仿真步和时间、渲染序号 |

`grasp_id` 仍是独立接触姿态编号的分组别名，不代表重新执行物理抓取；世界仿真时间在采样阶段保持不变，渲染序号和系统时间递增。训练只读取 `quality/sample_valid=True` 的行。

脚本（均相对于 `/data/nju/Lincao/InsertAnything_tactile`）：

- `scripts/collect_normal_angle_data.py` / `_collect_normal_angle_worker.py`：六物体采集与单任务运行。
- `source/tactile_pose/normal_sampling.py`：竖直、单轴角度定义与分层提议。
- `source/tactile_pose/tcp_kinematics.py`：从 USD 实际关节坐标系构建 FK/IK，使用有界 dogbox 求解。
- `source/tactile_pose/normal_sim_collection.py`：实际关节写入、世界竖直约束、同步记录。
- `source/tactile_pose/normal_learning.py`：单输出网络、数据集、分层划分和独立样本检查。
- `scripts/train_normal_angle.py` / `evaluate_normal_angle.py` / `predict_normal_angle.py`：训练、独立评估、单帧推理。
- `scripts/inspect_normal_angle_data.py`：独立标签验收和图集导出。

新旧模块共 **19 项测试通过**；原 InsertAnything 48 个保护文件和 UniVTAC 1,886 个保护文件未改变。已有 v3 接触渲染接口只新增兼容的采样字段入口，旧代码快照在 `validation/normal_v4_previous_sources_20261012`。早期 IK 未收敛尝试标为失败，不是训练源；本页仅使用名称包含 `train600_upright` 的完整新批次。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_pose.py tests/test_calibration.py tests/test_tcp_sampling.py tests/test_normal_sampling.py
./scripts/python.sh scripts/verify_isolation.py
```

历史流程：[v3 多接近方向采集](README_insertanything_tactile_tcp_v3.md)、[v2 抓取数据和三维旋转模型](README_insertanything_tactile_pose_v2.md)。
