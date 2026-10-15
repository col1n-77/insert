# InsertAnything v3：TCP 附近的独立接触姿态与双侧触觉数据

> 当前用户要求已改为固定插销世界竖直、夹爪绕夹持法向倾转、仅预测一个角度。请使用 [v4 单角度手册](README_insertanything_tactile_normal_v4.md)。本页保留此前多接近方向实验记录。

更新：2026-10-12。本页是新的**接触姿态数据生成入口**；代码位于 `/data/nju/Lincao/InsertAnything_tactile`。原 InsertAnything、UniVTAC、已有 v1/v2 数据与模型均保留。

## 1. 新流程与坐标定义

每次独立采样生成一对左右 RGB，不再将一次闭爪后连续保持的 8 帧当成 8 个姿态：

1. 读取物体网格在 HeldAsset 根坐标系中的包围盒，把**包围盒中心**放在真实 TCP 附近。物体根原点可能位于插销底部，不能直接将根原点当作几何中心。
2. 选择接近方向。默认共 5 个：沿物体 +Z，以及向物体 ±X、±Y 分别倾斜 20°。每个方向采集相同数量的有效样本，也可以直接提供方向向量。
3. 在 TCP 的 XY 平面均匀提议 ±2 mm 平移；这是垂直接近方向的平面。物体几何中心的 TCP Z 坐标默认 0。
4. 绕接近轴均匀提议 −180°～180° 的夹爪旋转。
5. 每对图像抽取一个共同的目标压入量，默认均匀提议 0.2～1.0 mm。独立调节左右手指位置，使**各侧深度图中的最大压入量**接近目标；不是要求整片胶垫处处具有相同深度。
6. 先检查全网格范围能否在关节限位内接触，再在相机视野内迭代调整，默认最多 12 次，误差容许 0.03 mm。每侧至少 64 个像素深于 0.02 mm；不满足者记为无效并重采。
7. 同步渲染左右 320×240 RGB，并读取物体、TCP 和手指的实际状态，保存完整位置与旋转真值。

真实 TCP 是 `panda_fingertip_centered` 刚体坐标系，TCP **+Z 是接近轴，±Y 是手指闭合方向**。`approach_directions_o` 表示 TCP +Z 在物体坐标系中的方向。绕轴角度描述夹爪旋转；物体相对 TCP 的旋转方向是其逆变换，不能直接把该角度当标签。

采用主动旋转、列向量，四元数为 **wxyz**：

```text
R_TO = R_WT.T @ R_WO
p_TO = R_WT.T @ (p_WO - p_WT)
point_T = R_TO @ point_O + p_TO
```

`p_TO` 是**物体根原点**在 TCP 中的位置；`object/bbox_center_t_m` 是几何中心在 TCP 中的位置。二者有意分别存储。世界坐标包含环境原点偏移。

注意当前胶垫感知面中心位于 TCP 的 Z≈−16.93 mm，默认几何中心仍严格放在 TCP 附近。因此倾斜姿态可能只在图像边缘接触。若研究目标是让物体几何中心对齐感知面中心，可以显式设置 `--center_z_mm -16.93`；这会改变位置分布，标签仍使用原真实 TCP。该选项尚未进行本页的六物体批量验收。

## 2. 0.2～1 mm 是否合适

可以作为本实现的**几何压入目标范围**，实际可达范围取决于物体、姿态和手指安装间隙。

- 胶体资产厚 4 mm；上游 `generic_gsmini_cfg.py` 标注最大压入参数 2 mm。本采集器限制请求不超过 2 mm，但该参数不是实物安全认证。
- [Taxim 官方示例](https://github.com/Robo-Touch/Taxim)使用 `-depth 1.0`，单位为 mm；这说明光学输入可以使用毫米级压入，并不证明当前安装下每个姿态都可实现 1 mm。
- 实测左右关节为 0 时，胶垫表面间隙约 **7.144 mm**。8 mm 圆插销在轴向姿态、可调整横向居中时，每侧最大共同压入约 `(8−7.144)/2 = 0.428 mm`。倾斜或换物体后范围不同，不能强迫关节越过下限来满足目标。
- 2026-10-12 对 L 插销的 5 个方向各补采 1 组：请求 0.2 mm 时实测 **0.19139～0.20001 mm**；请求 1 mm 时 **0.99363～0.99996 mm**。有效帧均无近裁切或胶厚饱和。

**提议分布均匀，接受后的分布不是均匀分布。** 深压入、过大横向偏移等不可达组合会被拒绝。每个方向的有效数量有配额；压入量、平移和绕轴角度没有强制均匀配额。统计训练数据覆盖率时需读取实际接受分布。

当前实现固定机器人姿态，直接写入物体与手指状态，然后 `sim.forward()` 和渲染；样本间不运行物理积分。使用刚体几何重叠驱动分侧校准的 Taxim 光学模型。**这些是静态接触观测，不是稳定抓取轨迹、力传感数据或软胶 FEM 形变真值。** 初始化会短暂运行原环境以得到有效的机器人状态；之后每个样本不执行抓取动作。

## 3. 采集命令

前提：现有 `/data/nju/Lincao/InsertAnything/.venv-isaac51/bin/python` 与 UniVTAC/TacEx、RTX GPU 可用。当前本机实际版本为 Isaac Sim 5.1 / Isaac Lab 2.3 / Python 3.11；不需重新安装环境。

六种插销，每种 100 个有效姿态，共 600 对图像。默认 5 个方向，每方向 20 个有效姿态；一个环境，六个任务串行启动独立 Isaac 进程以释放显存：

```bash
cd /data/nju/Lincao/InsertAnything_tactile
./scripts/python.sh scripts/collect_tcp_pose_data.py \
  --task all --samples_per_task 100 \
  --tilt_deg 20 --offset_xy_mm 2 2 \
  --roll_range_deg -180 180 \
  --depth_range_mm 0.2 1.0 \
  --seed 74000 --headless
```

默认创建带时间戳的新输出目录，结束打印 `OUTPUT`。可增加 `--output /绝对路径/新的输出目录`；已有目录会拒绝覆盖。`--samples_per_task` 必须是方向数的整数倍。`--max_attempts_per_sample 1000` 限制每个有效名额的提议次数，达到上限会明确失败并保留已有记录，避免无限重试。

只采 L 插销，用 `--task LHole-III`；可选任务名为 `CircleHole-I`、`SquareHole-II`、`LHole-III`、`TriangleHole-IV`、`HexagonHole-III`、`HexagonHole-IV`。

自定义三个接近方向的示例（本页批量实测使用默认五方向；自定义入口通过数学测试，以下分布尚未批量采集）：

```bash
./scripts/python.sh scripts/collect_tcp_pose_data.py \
  --task LHole-III --samples_per_task 90 \
  --approach_directions "0,0,1;0.5,0,0.8660254;-0.5,0,0.8660254" \
  --offset_xy_mm 2 2 --depth_range_mm 0.2 1.0 \
  --seed 75000 --headless
```

多个向量放在同一个引号内，用分号分隔，向量自动归一化。`--approach_directions` 覆盖 `--tilt_deg`。以上 100/90 样本命令为扩量命令，本轮只运行下方列出的验证批次。

## 4. 新生成的数据与实测结果

主验证目录：

`/data/nju/Lincao/InsertAnything_tactile/outputs/tcp_pose_v3_validation_20261012`

参数：`--task all --samples_per_task 10 --seed 71000 --headless`，其余为默认值。

| 插销 | 有效姿态 | 已渲染尝试 | 全部提议 | 有效帧双侧压入范围 mm |
| --- | ---: | ---: | ---: | --- |
| CircleHole-I | 10 | 108 | 320 | 0.234～0.765 |
| SquareHole-II | 10 | 18 | 19 | 0.208～0.986 |
| LHole-III | 10 | 11 | 11 | 0.256～0.827 |
| TriangleHole-IV | 10 | 42 | 46 | 0.266～0.959 |
| HexagonHole-III | 10 | 15 | 15 | 0.211～0.990 |
| HexagonHole-IV | 10 | 18 | 19 | 0.340～0.972 |

共 430 次提议，218 次在渲染前被几何限位筛除，212 次已渲染记录中有 60 次有效。每物体每方向 2 次有效；每侧有效 RGB 均为 10 张不同图像。双侧压入最大目标误差 **0.02700 mm**；相对位姿与从世界坐标重新计算结果的最大位置差小于 `1e-9 m`、旋转矩阵元素差小于 `3e-8`。这些只是标签一致性验收，不代表网络预测误差。

文件结构：

```text
tcp_pose_v3_validation_20261012/
  config.json                          # 本批参数
  collection.json                      # 六任务状态、数据 UUID 和 HDF5 路径
  CircleHole-I.log                      # 每任务完整仿真日志
  CircleHole-I/
    metadata.json                      # 坐标、标定、资产、限位、状态与统计
    samples.h5                         # 所有已渲染尝试，包括无效帧
    proposals.jsonl                    # 每次提议的种子、参数、拒绝原因/HDF5 行号
    direction_0_left.png                # 每方向第一个有效样本的原始 RGB
    direction_0_right.png
    ...
  SquareHole-II/ ...                    # 其余五物体结构相同
  inspection/
    validation.json                    # 六物体离线验收
    stereo_overview.png                # 六物体左右图像总览
    LHole-III_directions.png            # 每物体五方向图集
    ...
```

[六物体双侧触觉总览](/data/nju/Lincao/InsertAnything_tactile/outputs/tcp_pose_v3_validation_20261012/inspection/stereo_overview.png) · [L 插销五方向](/data/nju/Lincao/InsertAnything_tactile/outputs/tcp_pose_v3_validation_20261012/inspection/LHole-III_directions.png) · [验收 JSON](/data/nju/Lincao/InsertAnything_tactile/outputs/tcp_pose_v3_validation_20261012/inspection/validation.json)

端点验证保存在 `outputs/tcp_pose_v3_depth02_20261012`（seed 72000）和 `outputs/tcp_pose_v3_depth10_20261012`（seed 73000），每批 5 组有效 L 插销姿态。命令为单任务 `--samples_per_task 5`，分别指定 `--depth_range_mm 0.2 0.2` / `--depth_range_mm 1.0 1.0`。各批均有自己的 `inspection/validation.json`。

## 5. HDF5 字段与有效样本筛选

Schema 为 `tactile_tcp_pose_v3`，所有数据按第一维行号同步。无效的已渲染尝试也保留左右图像和实际位姿；渲染前筛除的提议只有 JSONL 记录，不伪造图像。

| 字段 | 内容 |
| --- | --- |
| `tactile/{left,right}/rgb` | 原生方向 uint8 RGB，N×240×320×3 |
| `tactile/{left,right}/press_depth_mm` | 非负几何压入图，N×240×320 |
| `tactile/{left,right}/camera_depth_mm` | 图像平面深度，非有限空射线替换为远裁面 |
| `gripper/finger_joint_position_m` | 左右实际关节位置，N×2 |
| `gripper/opening_joint_sum_m` | 两关节位移之和，N×1；不等于胶垫表面间隙 |
| `gripper/gel_front_gap_m` | 加上实测安装偏移后的两表面间距 |
| `tcp/position_w_m`、`tcp/quaternion_wxyz` | 实际 TCP 世界位姿 |
| `object/position_w_m`、`object/quaternion_wxyz` | 实际物体根世界位姿 |
| `target/position_to_m` | 物体相对 TCP 的实际平移，N×3 |
| `target/rotation_to`、`target/quaternion_to_wxyz` | 实际相对旋转矩阵 N×3×3、四元数 N×4 |
| `object/bbox_center_t_m` | 实际几何中心相对 TCP 的位置 |
| `sampling/*` | 方向、绕轴角、平面偏移、目标旋转/中心、目标压入、调整次数 |
| `contact/*` | 每侧接触像素数、最大压入、目标误差、有效性、裁切像素数 |
| `sensor/{side}_camera_*` | 由运行时传感器刚体 × 静态相机外参计算的实际相机位姿 |
| `robot/joint_position`、`robot/joint_velocity` | 全部实际关节状态；静态样本速度置零 |
| `ids/*` | 物体、环境、方向、名额、重试、提议/接触姿态、HDF5 行、有效样本编号 |
| `random/seed` | 此独立提议的种子；根种子和初始化种子在 metadata |
| `time/unix_ns` | 采样时的系统时间，纳秒 |
| `time/sim_time_s`、`time/physics_step` | 真实仿真时间/步号；静态采样中保持不变 |
| `time/render_refresh_index` | 递增渲染刷新编号，一次刷新含两次渲染 |
| `quality/sample_valid`、`quality/status_code` | 有效掩码与失败原因位标志 |

`grasp_id` 仅作为 `contact_pose_id` 的分组兼容别名，**本流程没有独立物理抓取回合**。使用 `dataset_uuid + contact_pose_id` 才能跨文件唯一标识姿态；不同任务的局部 ID 会重复。提议的种子还由稳定 object_id、direction_id、slot_id、retry_id 派生。

读取有效样本示例：

```python
import h5py
import numpy as np

path = "/data/nju/Lincao/InsertAnything_tactile/outputs/tcp_pose_v3_validation_20261012/LHole-III/samples.h5"
with h5py.File(path, "r") as f:
    rows = np.flatnonzero(f["quality/sample_valid"][:])
    i = int(rows[0])
    left = f["tactile/left/rgb"][i]
    right = f["tactile/right/rgb"][i]
    fingers = f["gripper/finger_joint_position_m"][i]
    p_to = f["target/position_to_m"][i]
    r_to = f["target/rotation_to"][i]
```

质量位：1 左侧无有效接触；2 右侧无有效接触；4 非有限数；8 未达到目标深度；16 关节越限；32 深度裁切或饱和；64 写入与实测位姿不一致。多种原因可同时置位；只用 `quality/sample_valid == True` 的行训练。

## 6. 查看、核验与训练边界

离线验收命令无需启动 Isaac Sim，输入可以是批次目录、单任务目录或 HDF5；输出目录必须为新目录：

```bash
cd /data/nju/Lincao/InsertAnything_tactile
./scripts/python.sh scripts/inspect_tcp_pose_data.py \
  /data/nju/Lincao/InsertAnything_tactile/outputs/tcp_pose_v3_validation_20261012 \
  --output /tmp/insertanything_tcp_v3_review
```

本页 `inspection` 已通过验收。采样方向、坐标变换、种子独立性连同已有校准/网络测试共 **14 项通过**。原项目隔离检查的 48 个 InsertAnything 文件和 1,886 个 UniVTAC 文件均未变化。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_pose.py tests/test_calibration.py tests/test_tcp_sampling.py
./scripts/python.sh scripts/verify_isolation.py
```

关闭第三方 pytest 插件自动加载，避免当前终端 ROS Python 3.12 插件污染 Python 3.11 测试环境；没有修改任何运行环境。

本轮完成采样器、数据、标签和预览，未用 v3 重训网络。旧训练入口明确只接受 v1/v2，因此会拒绝 v3，而不会把 TCP 标签与 `panda_hand` 标签混合。接入训练时需要读取 `target/rotation_to`，记录目标坐标系及新分布；平移标签也已保存，但旧网络仅预测旋转。六物体混训还需要物体条件或分别训练；圆形及规则多边形存在几何对称导致的姿态歧义，应据此设计角度损失/评估。

旧抓取采集、网络与已有模型见 [v2 运行手册](README_insertanything_tactile_pose_v2.md)。

实现：`scripts/collect_tcp_pose_data.py`（任务调度/参数）、`scripts/_collect_tcp_pose_worker.py`（单任务采集）、`source/tactile_pose/tcp_sampling.py`（采样数学）、`source/tactile_pose/tcp_sim_collection.py`（运行时渲染与写入）、`scripts/inspect_tcp_pose_data.py`（离线核验）。均位于 `/data/nju/Lincao/InsertAnything_tactile`。
