# InsertAnything：独立 GelSight Mini 触觉扩展

更新：2026-10-11。代码目录：`/data/nju/Lincao/InsertAnything_tactile`。

2026-10-12 新增：[双侧触觉＋夹爪开度的物体相对旋转估计](README_insertanything_tactile_pose.md)，包含独立抓取采集、实际旋转真值、CNN/MLP 训练与推理。下文仍是原策略评估及触觉录像入口。

2026-10-12 已渲染：[全部六场景、夹爪/孔位近景及同步左右触觉图集](README_insertanything_tactile_scenes.md)，每个场景均为 seed 0、初始化后保持 8 步的快照；包含各侧接触状态标注。

2026-10-12 已将[左右零点诊断](README_insertanything_tactile_symmetry.md)的结果接入实际运行：默认按左 28.246235 mm、右 27.950882 mm 生成高度图及 RGB，空载与对称物体验证通过。历史录像/数据保留原 28 mm 基准；新采集和录像记录校准版本。见[分侧校准与重新采集训练手册](README_insertanything_tactile_pose_v2.md)。

## 当前实现与隔离方式

- 新目录与 `/data/nju/Lincao/InsertAnything` 同级。原 InsertAnything 和 UniVTAC 的源码、任务配置及权重不作修改。
- 通过 Python 继承原任务、注册新的 `InsertAnythingTactile-*` ID 实现；运行时创建各自的配置副本。原 `InsertAnything-*` 命令继续使用原夹爪。
- 使用 UniVTAC/TacEx 的 **GelSight Mini + GPU Taxim** 光学模型。新任务的 Franka 夹爪两侧都有传感器资产，可选择只读取左侧、右侧或两侧。
- 物理接触仍由 **PhysX 刚体胶垫**计算；深度相机的接触几何送入 Taxim 生成 RGB。这不是完整 UIPC/FEM 软胶形变，没有剪切标记点模型，也不把几何压入深度当成力或压力。
- 触觉图像作为独立接口，不拼入原策略观测。已有 L 孔、六角孔 III 权重可加载，但不会利用触觉 RGB 做决策。融合触觉的策略需要另行设计输入并训练。
- 不复制大模型和运行环境。启动脚本复用 `/data/nju/Lincao/InsertAnything/.venv-isaac51/bin/python`，并仅向当前进程添加新目录的 `source` 路径。已验证 Python 3.11、Isaac Sim 5.1、Isaac Lab 2.3、TacEx 环境。
- 输出保存在新项目的 `outputs/`；已在新配置中关闭原任务会写回原目录的力 CSV logger。

## 完整评估录像与双侧触觉记录

2026-10-11 已实测：L 孔原权重、seed 0，完成 **1 回合，成功 1 次，首次成功 7.47 s**。录制了 **299 帧，15 Hz，19.93 s**；视频与全部 46 个 HDF5 数据集、299 行状态 CSV 逐帧对齐，所有数值有限，末帧在自动重置前保存。单回合结果不能代表大样本成功率。

无需激活 Conda，以下第二行完整复制即可，不需要反斜杠续行：

```bash
cd /data/nju/Lincao/InsertAnything_tactile
./scripts/python.sh scripts/record_eval.py --task InsertAnythingTactile-LHole-III-Direct-v0 --num_envs 1 --seed 0 --headless
```

- 自动加载原目录的 `InsertAnything-LHole-III-Direct-v0.pth`，始终读取左右两个 GelSight Mini；触觉 RGB 不进入已有策略的动作输入。
- 自动完成一个完整回合后退出；此录制入口只支持 `--num_envs 1`。沿用原任务超时规则 `episode_length >= max_episode_length - 1`，因此 L 孔配置 300 步，录制实际为 299 步。
- 自动创建 `outputs/eval_时间戳/`。可追加 `--output outputs/新名称`，已有目录会被拒绝，避免覆盖历史数据。去掉 `--headless` 可请求 GUI；本次完整录像验证使用无界面模式。
- 可用 `--checkpoint /绝对路径/策略.pth` 指定兼容权重。改为 `--task InsertAnythingTactile-HexagonHole-III-Direct-v0` 时，会寻找对应原六角孔 III 权重；其完整录制尚未实测。其他任务没有默认权重时会在启动仿真前报错。
- 依赖当前已有的 `h5py`、`imageio`、`imageio-ffmpeg`。离线查看脚本另外使用当前已有的 OpenCV 和 matplotlib。

本次实测目录：`/data/nju/Lincao/InsertAnything_tactile/outputs/eval_lhole_stereo_20261011_01/`。运行日志：`/data/nju/Lincao/InsertAnything_tactile/logs/eval_lhole_stereo_20261011_01.log`。

| 文件 | 内容 |
|---|---|
| `scene.mp4` | 外部场景录像，960×720，15 FPS |
| `left_tactile.mp4`、`right_tactile.mp4` | 左右触觉 RGB 预览，320×240，15 FPS |
| `trajectory.h5` | 无损逐帧触觉 RGB、float32 深度及全部数值轨迹 |
| `states.csv` | 接触力、9 关节位置/速度、控制器 effort 指令、末端位姿/速度、插销/孔位姿、动作、奖励、成功及终止标志；不含图像和观测向量 |
| `reset.npz` | 回合开始前的双侧 RGB、压入深度、相机深度快照 |
| `first_*.png`、`terminal_*.png` | 录制第一步与重置前最后一步的场景、左右触觉快照 |
| `report.json` | 权重路径和 SHA256、单位、配置、成功时间、采样频率、帧数等；完整运行应为 `status: complete` |
| `validation.json`、`preview.png`、`signals.png` | 离线检查脚本生成的验收结果、场景/双侧 RGB/深度对照、力/末端位移/触觉深度曲线 |

HDF5 的主要字段（`T=299` 为本次样本数）：

- `tactile/left/rgb`、`tactile/right/rgb`：`[T,240,320,3] uint8`。MP4 是有损预览，做数据分析或训练请读这些原始数组。
- `tactile/{left,right}/press_depth_mm`、`camera_depth_mm`：`[T,240,320] float32`，单位 mm。压入深度是几何代理量，不是力或压力。
- `force/contact_local_n`：`[T,3]`，原任务使用的插销坐标系接触力，经过去皮与 EMA；`force/net_normal_world_n` 是插销上各接触体的世界坐标系净法向力之和，不包含切向力，也不是仅筛选夹爪—插销接触对的力。未提供双侧传感器各自的力标定值。
- `robot/joint_position`、`joint_velocity`、`commanded_joint_effort`：`[T,9]`；关节顺序和单位见 `report.json`。前 7 项为机械臂旋转关节，后 2 项为夹爪直线关节。effort 是控制器指令，不是测量扭矩。
- `robot/ee_position_env_m`、`ee_quaternion_wxyz`、`ee_linear_velocity_w_m_s`、`ee_angular_velocity_w_rad_s`：末端状态。位置相对环境原点，速度为世界坐标轴分量，四元数顺序为 wxyz。
- `policy/observation_before_action`、`input_after_clip`、`critic_observation_before_action`、`action`、`smoothed_action`：原始观测、裁剪后策略输入、critic 状态、执行动作及 EMA 动作；策略内部还会按 checkpoint 归一化。
- `step`、`time_s`、`reward`、`terminated`、`truncated`、`metrics/*`、`reward_terms/*`：时序、回报、回合标志和阶段指标。

**对齐约定：**第 i 行是第 i+1 次动作前的策略输入/动作，以及该动作执行后的机械臂/物体状态、力、触觉图像、奖励与终止标志。`time_s` 从策略开始执行计时，不包括初始化抓取及回合重置的内部仿真步。录制回调在 `_get_rewards()` 完成后、自动重置前采样，避免最后一帧混入下一回合初始状态。

离线检查并生成预览与曲线，不启动 Isaac Sim：

```bash
cd /data/nju/Lincao/InsertAnything_tactile
./scripts/python.sh scripts/inspect_recording.py outputs/eval_lhole_stereo_20261011_01
```

读取一帧原始数据：

```python
import h5py
path = "/data/nju/Lincao/InsertAnything_tactile/outputs/eval_lhole_stereo_20261011_01/trajectory.h5"
with h5py.File(path, "r") as f:
    left_rgb = f["tactile/left/rgb"][0]
    right_depth_mm = f["tactile/right/press_depth_mm"][0]
    joint_positions = f["robot/joint_position"][:]
    contact_force_n = f["force/contact_local_n"][:]
```

## 快速接触检查与快照

以下命令无需激活 Conda。先使用 1 个环境验证。默认记录双侧触觉；`--sensors left` 或 `--sensors right` 只启用对应侧的相机和 Taxim 运算。

保持初始化后的夹持姿态，执行 60 步，并进行移除插销的接触对照：

```bash
cd /data/nju/Lincao/InsertAnything_tactile
./scripts/python.sh scripts/capture.py \
  --task InsertAnythingTactile-LHole-III-Direct-v0 \
  --num_envs 1 --steps 60 --seed 0 --headless --check_contact
```

使用原 L 孔权重执行 300 步（15 Hz，约一个任务回合）：

```bash
cd /data/nju/Lincao/InsertAnything_tactile
./scripts/python.sh scripts/capture.py \
  --task InsertAnythingTactile-LHole-III-Direct-v0 \
  --num_envs 1 --steps 300 --seed 0 --headless \
  --checkpoint /data/nju/Lincao/InsertAnything/source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/InsertAnything-LHole-III-Direct-v0.pth
```

六角孔 III、两个并行环境、仅左侧传感器：

```bash
cd /data/nju/Lincao/InsertAnything_tactile
./scripts/python.sh scripts/capture.py \
  --task InsertAnythingTactile-HexagonHole-III-Direct-v0 \
  --num_envs 2 --sensors left --steps 30 --seed 0 --headless --check_contact \
  --checkpoint /data/nju/Lincao/InsertAnything/source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/InsertAnything-HexagonHole-III-Direct-v0.pth
```

`capture.py` 保存起点/终点快照，不保存完整录像和逐步轨迹；完整录制请用上面的 `record_eval.py`。脚本自动启用相机。去掉 `--headless` 可请求 GUI；用户已提供 GUI 启动成功截图，本机自动验收使用无界面模式。全部运行都有步数上限。可用 `--output outputs/自定名称` 指定输出，但目录必须尚不存在，以免覆盖历史记录。新夹爪目前只检查了 1、2 个并行环境，未验证 128 环境渲染规模。

## 任务与配置位置

配置文件：[config.py](/data/nju/Lincao/InsertAnything_tactile/source/insertanything_tactile/config.py)。

| 新任务 ID | 独立配置类 | 本次运行验证 |
|---|---|---|
| `InsertAnythingTactile-CircleHole-I-Direct-v0` | `CircleHoleCfg` | 配置检查、双侧 8 步快照 |
| `InsertAnythingTactile-SquareHole-II-Direct-v0` | `SquareHoleCfg` | 配置检查、双侧 8 步快照 |
| `InsertAnythingTactile-LHole-III-Direct-v0` | `LHoleCfg` | 双侧接触检查、原权重完整回合、双侧 8 步快照 |
| `InsertAnythingTactile-TriangleHole-IV-Direct-v0` | `TriangleHoleCfg` | 配置检查、双侧 8 步快照 |
| `InsertAnythingTactile-HexagonHole-III-Direct-v0` | `HexagonHoleIIICfg` | 2 环境原权重左侧 30 步、双侧 8 步快照 |
| `InsertAnythingTactile-HexagonHole-IV-Direct-v0` | `HexagonHoleIVCfg` | 配置检查、双侧 8 步快照 |

文件用途：

- `source/insertanything_tactile/config.py`：独立环境配置，`GelSightMiniCfg` 中可修改读取侧、分辨率、相机裁剪范围；默认 320×240、24–34 mm。任务几何、奖励和控制器继承原实现。
- `source/insertanything_tactile/env.py`：创建 `GelSightSensor`、设置 Taxim、提供 `get_tactile_observations()` 和 `refresh_tactile()`。
- `source/insertanything_tactile/__init__.py`：注册上述六个独立任务 ID。
- `source/insertanything_tactile/assets.py`：生成新目录内的 USD 组合文件。TacEx 刚体资产原来引用缺失的 `Gelpad_low_res.usd`，这里仅在本地副本中引用现有 `Gelpad_mid_res.usd`，并为胶垫及虚拟指尖设置有效质量。
- `assets/franka_gsmini_physx.usda`：新任务实际加载的机器人资产；网格、材质继续引用 UniVTAC。不会回写被引用资产。
- `source/insertanything_tactile/policy.py`：读取已有 RL-Games 权重，沿用原观测裁剪和动作裁剪。
- `source/insertanything_tactile/recording.py`：录制专用环境子类，在自动重置前同步采样，写入 HDF5、CSV 和三个视频。
- `scripts/record_eval.py`：加载已有权重，执行一个完整回合，记录双侧触觉和机械臂状态。
- `scripts/inspect_recording.py`：离线验证视频、HDF5 和 CSV，并生成预览和曲线。
- `scripts/capture.py`：有限步数运行、保存图像及数组、接触移除对照。
- `scripts/plot_capture.py`：将接触 RGB、几何压入深度和移除插销后的 RGB 绘成对照图。

## 快照输出与程序接口

默认输出目录是 `/data/nju/Lincao/InsertAnything_tactile/outputs/时间戳/`。

- `reset.npz`、`contact.npz`：初始化后与运行结束时的所有环境数据。`contact` 是结束时快照，未开启接触对照时不保证该时刻仍有接触；不是完整轨迹数据集。
- `reset_left.png`、`contact_left.png` 等：第 0 个环境各侧的 RGB。
- `scene.png`：第 0 个环境的外部场景截图。
- `peg_removed.npz` 和相应 PNG：仅在 `--check_contact` 下保存。
- `report.json`：张量形状、深度范围、策略来源、完成回合数、接触检查结果。验收应查看 `status: passed`。

```python
# env 是 gym.make 创建的新触觉任务
tactile = env.unwrapped.get_tactile_observations(clone=True)
left_rgb = tactile["left"]["rgb"]                    # [N, 240, 320, 3], uint8
left_press = tactile["left"]["press_depth_mm"]       # [N, 240, 320], float32, mm
left_depth = tactile["left"]["camera_depth_mm"]      # [N, 240, 320], float32, mm
```

当前默认配置下 `press_depth_mm = clip(zero_side_mm - camera_depth_mm, 0, 4)`，左零点 28.246235 mm、右零点 27.950882 mm；历史 `legacy_shared` 模式共用 28 mm。是光学模型的几何压入代理量，不是校准后的压力值。无命中像素的相机深度用远裁剪面 34 mm 表示。NPZ 键为 `left_rgb`、`left_press_depth_mm`、`left_camera_depth_mm`，右侧同理。返回值在 GPU 上；需要长期保存时使用 `clone=True` 或立即拷贝 CPU，避免后续仿真覆盖缓冲区。

绘制已有接触验证结果：

```bash
cd /data/nju/Lincao/InsertAnything_tactile
./scripts/python.sh scripts/plot_capture.py outputs/contact_validation
```

## 本次实测证据

- `outputs/contact_validation/report.json`：L 孔、1 环境、双侧、60 步，RGB 均为 `[1,240,320,3] uint8`；移除插销前左/右最大压入约 0.096/0.410 mm，移除后均为 0，RGB 平均绝对变化约 1.84/2.57（0–255 灰度单位）。
- `outputs/lhole_policy_1episode/report.json`、`logs/lhole_policy_1episode.log`：原 L 孔权重，seed 0，300 步，完成 1 回合并成功 1 次，首次成功时间 7.47 s。运行期间左右最大压入约 0.622/1.000 mm。这是兼容性试跑，不能代替原夹爪的 128 环境成功率评估。
- `outputs/hexagon_parallel_left/report.json`：原六角孔 III 权重，2 环境、仅左侧、30 步；观测 `[2,26]`、critic `[2,73]`，两个环境移走插销后压入均归零且 RGB 变化。没有完成整个六角孔回合。
- `validation/original_source_hashes.json`：接入前的源码/配置校验快照，覆盖 InsertAnything 48 个文件和 UniVTAC 1,886 个文件。校验命令如下；它核对已有文件内容，不删除新文件、不恢复文件。

```bash
cd /data/nju/Lincao/InsertAnything_tactile
./scripts/python.sh scripts/verify_isolation.py
./scripts/python.sh scripts/check_configs.py --headless
```

新任务六个配置都检查为独立副本，创建新配置不会更改原任务配置。最早两次开发探测没有通过接触检查，保留在 `outputs/first_contact_probe`、`outputs/contact_probe_v2`；当前验收以 `contact_validation` 及上述策略运行结果为准。
