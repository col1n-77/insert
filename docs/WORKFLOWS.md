# 安装、训练、评估与数据操作

[返回 README](../README.md) · [匹配奖励实验](MATCHED_REWARD_STUDY.md)

以下命令从仓库根目录执行；占位的绝对路径需要替换。扩展启动器自动切换到对应项目目录，所以扩展的相对 `outputs/` 位于扩展内。为串联采集、训练和评估，示例使用绝对路径变量及新的输出名。

## 环境与短检查

先在独立 Python 3.11 环境安装 Isaac Sim / Isaac Lab、Isaac Lab RL / Tasks、RL-Games、PyTorch、toml 和 wheel。工作站版本见 README。复用已有安装：

```bash
export INSERT_REPO="$PWD"
export INSERTANYTHING_BASE_PYTHON=/absolute/path/to/isaaclab-env/bin/python
bash scripts/setup_local_env.sh
./scripts/python.sh -m pip check
./scripts/python.sh scripts/list_envs.py
```

`setup_local_env.sh` 创建带 system-site-packages 的根目录 `.venv-isaac51`，并以 editable 模式安装原版包。已有环境与指定 Python 不匹配时会报错并保留环境，不会覆盖。或者设置 `INSERTANYTHING_PYTHON=/absolute/path/to/python` 让三个启动器直接复用完整环境。

原版/Angle 不要求 TacEx。有限仿真检查：

```bash
./scripts/python.sh scripts/smoke_test.py \
  --task InsertAnything-LHole-III-Direct-v0 \
  --num_envs 4 --steps 100 --headless --report outputs/base_smoke.json

./scripts/python.sh scripts/check_inhand_6d.py \
  --num_envs 4 --headless --report outputs/inhand_6d.json

./InsertAnything_angle/scripts/python.sh scripts/check_angle_env.py \
  --num_envs 128 --steps 8 --headless \
  --output "outputs/angle_check_$(date +%Y%m%d_%H%M%S).json"
```

这些检查会启动仿真，但不是训练。CPU 检查见 [VALIDATION.md](VALIDATION.md)。

## RL 训练

推荐的双组对照命令与参数解释见 [README](../README.md#4-当前推荐的奖励对照训练) 和 [MATCHED_REWARD_STUDY.md](MATCHED_REWARD_STUDY.md)。两组队列固定 LHole-III；`--plan_only` 不启动子进程，正式训练需去掉该参数。

单个 Angle 训练和原版 ±10°/六自由度训练命令也在 README。几点容易混淆的设置：

- 原版不加覆盖参数时默认偏角 0°、锁 roll/pitch；Angle 默认 ±10°、解锁。
- 单次 Angle 入口默认 constant；当前匹配队列默认 adaptive，推荐显式传调度。
- `--align_weight 0` 仅关闭新增轴线项，仍有角度输入，原力惩罚保留。
- `--depth_axis_gate` 默认关闭，只对单独研究门控的训练显式开启。
- `--num_envs × --horizon` 必须能整除 minibatch，horizon 和 minibatch 都需能整除 sequence length。
- `max_iterations` 为 PPO 轮次，不是回合数。环境数、minibatch 和 mini_epochs 共同决定每轮优化量。

检查 TensorBoard 中的学习率和训练曲线：

```bash
./scripts/python.sh -m tensorboard.main \
  --logdir "$INSERT_REPO/InsertAnything_angle/logs/rl_games/InsertAnythingAngle" \
  --port 6006
```

## 评估输出与动作模式

Angle 权重需要所属训练目录中的 `params/env.yaml`、`params/agent.yaml`、`run.json`。填入实际路径：

```bash
export INSERT_ANGLE_RUN=/absolute/path/to/angle/training/run
export INSERT_ANGLE_CHECKPOINT="$INSERT_ANGLE_RUN/nn/your_checkpoint.pth"
./InsertAnything_angle/scripts/python.sh scripts/eval_angle_policy.py \
  --checkpoint "$INSERT_ANGLE_CHECKPOINT" --training_run "$INSERT_ANGLE_RUN" \
  --num_envs 128 --seeds 1001 2002 --modes deterministic stochastic \
  --headless --output "outputs/manual_eval_$(date +%Y%m%d_%H%M%S)"
```

录制选定环境，只需追加 `--video --video_env 0`；这不会使其他环境自动退出评估。脚本会开启相机。每种模式/种子执行一个完整原生回合/环境，记录在动作物理执行后、自动 reset 前。

| 文件 | 用途 |
| --- | --- |
| `report.json` | 权重哈希、epoch、配置、回合成功率及误差汇总 |
| `evaluated_checkpoint.pth` | 本次实际评估的冻结副本 |
| `training_env.yaml`、`training_agent.yaml`、`training_run.json` | 保存训练契约 |
| `effective_env.yaml` | 本次实际评估配置 |
| `<mode>_seed_<seed>/states.npz` | 完整逐步数组：状态、原始/裁剪/EMA 动作、奖励、误差、力等 |
| 同目录 `diagnostics.csv`、`summary.json` | 逐步 CSV、汇总及逐回合诊断 |
| 同目录视频/成功帧 | 仅开启录像或满足对应成功条件时生成 |

`deterministic` 使用策略均值，`stochastic` 从策略分布采样。保存的实际动作还会经过标准裁剪和控制器处理。相同随机种子用于控制比较条件；新版本是否完全相同初始化，可检查逐步状态的初始帧。

原版六自由度权重使用原版入口；该脚本检查保存配置必须是 ±10°、roll/pitch 解锁：

```bash
export INSERT_BASE_RUN=/absolute/path/to/base/training/run
export INSERT_BASE_CHECKPOINT="$INSERT_BASE_RUN/nn/your_checkpoint.pth"
./scripts/python.sh scripts/eval_inhand_policy.py \
  --checkpoint "$INSERT_BASE_CHECKPOINT" --training_run "$INSERT_BASE_RUN" \
  --num_envs 128 --seeds 1001 2002 --modes deterministic stochastic \
  --headless --output "outputs/base_eval_$(date +%Y%m%d_%H%M%S)"
```

原版上游 0°/锁定姿态的演示使用 `scripts/rl_games/play.py` 和匹配的上游权重，不能用上面的 ±10° 评估入口强行代替。

## 触觉的额外依赖

需在选定 Isaac Python 中准备 `tacex`、`tacex_assets`，以及其 GelSight Mini 机器人/胶垫网格和 Taxim 光学标定。已有 UniVTAC 集成环境可提供这些依赖；本仓库不包含该外部安装及资产，也不自动下载。数据处理/绘图/视频还需要 h5py、SciPy、Pillow、Matplotlib、imageio、imageio-ffmpeg、OpenCV。

通过模块定位检查 TacEx 安装（不启动仿真）：

```bash
./scripts/python.sh - <<'PY'
import importlib.util
for name in ('tacex', 'tacex_assets', 'h5py', 'scipy', 'PIL', 'matplotlib', 'imageio', 'cv2'):
    spec = importlib.util.find_spec(name)
    print(name, spec.origin if spec else 'MISSING')
PY
```

首次创建环境时，`prepare_robot_asset()` 从已安装资产生成本机 `InsertAnything_tactile/assets/franka_gsmini_physx.usda`。该层引用外部网格，不应跨机器直接复用；重定位方法见[资产说明](../InsertAnything_tactile/assets/README.md)。该模拟采用刚体胶垫和 Taxim RGB，不含 UIPC 形变或剪切标记模型。

## 双触觉检查和评估录像

```bash
./InsertAnything_tactile/scripts/python.sh scripts/capture.py \
  --task InsertAnythingTactile-LHole-III-Direct-v0 \
  --num_envs 1 --steps 60 --check_contact --headless \
  --output "outputs/contact_check_$(date +%Y%m%d_%H%M%S)"
```

未给 checkpoint 时使用零动作保持，`status=passed` 表示接触/渲染检查通过，不等于策略完成插入。输出 reset/contact/移除插销后的双侧 PNG、NPZ 与报告。

用兼容原版观测的策略录制完整回合，附双触觉和状态：

```bash
export INSERT_RECORD_CHECKPOINT=/absolute/path/to/compatible/base_policy.pth
export INSERT_RECORD_OUT="$INSERT_REPO/InsertAnything_tactile/outputs/record_$(date +%Y%m%d_%H%M%S)"
./InsertAnything_tactile/scripts/python.sh scripts/record_eval.py \
  --task InsertAnythingTactile-LHole-III-Direct-v0 \
  --checkpoint "$INSERT_RECORD_CHECKPOINT" --num_envs 1 --seed 0 \
  --headless --output "$INSERT_RECORD_OUT"
./InsertAnything_tactile/scripts/python.sh scripts/inspect_recording.py "$INSERT_RECORD_OUT"
```

原始 RGB/深度和状态保存在 `trajectory.h5`，视频为有损预览。关节 effort 是控制命令，不是测得力矩；`press_depth_mm` 是几何压入代理，不是压力。只采集触觉的旧任务不会把 RGB 输入原策略，也不等同于 `InsertAnythingTactile6D-*`。

## 单角度数据采集与训练

v4 流程固定插销竖直、只改变绕夹持法向的相对角度。把物体放在 TCP 中心附近，解算夹爪姿态，随机横向偏移和几何压入，渲染后读取实际位姿。采样时不推进接触动力学到新的抓取回合；`grasp_id` 是独立接触姿态编号的别名。世界仿真时间可保持不变，系统时间与渲染序号递增。

六物体各 100 组有效样本，角度范围 ±15°、5 个区间等额采样：

```bash
export INSERT_DATA_TRAIN="$INSERT_REPO/InsertAnything_tactile/outputs/normal_train_$(date +%Y%m%d_%H%M%S)"
./InsertAnything_tactile/scripts/python.sh scripts/collect_normal_angle_data.py \
  --task all --samples_per_task 100 --angle_range_deg -15 15 --angle_bins 5 \
  --offset_xy_mm 2 2 --depth_range_mm 0.2 1.0 --seed 81000 --headless \
  --output "$INSERT_DATA_TRAIN"
```

`0.2～1 mm` 是当前几何光学代理使用的采样范围，不是实际 GelSight 硬件的受力/安全规格。深度误差、接触像素、裁剪饱和、关节界限、竖直和单轴约束均参与有效性筛选；HDF5 可保留无效尝试，训练只读 `sample_valid=True`。

每个物体子目录含 `samples.h5`、`metadata.json` 及预览，顶层 `collection.json` 索引必须为 complete。主要数据字段：

| 字段 | 含义/单位 |
| --- | --- |
| `tactile/{left,right}/rgb` | uint8，原生 240×320×3 |
| `tactile/{left,right}/press_depth_mm`、`camera_depth_mm` | 几何压入和相机深度，mm |
| `gripper/finger_joint_position_m`、`opening_joint_sum_m` | 两指实测位移及其和，m |
| `gripper/gel_front_gap_m` | 另含安装偏移的胶面间距，m |
| `tcp/position_w_m`、`tcp/quaternion_wxyz` | TCP 世界位姿 |
| `object/position_w_m`、`object/quaternion_wxyz` | 物体世界位姿 |
| `target/position_to_m`、`rotation_to`、`quaternion_to_wxyz` | 物体相对 TCP 的实际完整位姿 |
| `target/in_hand_angle_deg` | 唯一预测目标，绕 TCP +Y 有符号偏角，度 |
| `time/*`、`ids/*`、`random/seed` | 系统时间、仿真步/渲染序号、物体/环境/接触姿态编号、种子 |
| `contact/*`、`quality/*` | 双侧有效性、像素数、深度误差、单轴/竖直残差、状态码 |

仅用左右 RGB 和关节位移和训练：

```bash
export INSERT_ANGLE_MODEL="$INSERT_REPO/InsertAnything_tactile/outputs/normal_model_$(date +%Y%m%d_%H%M%S)"
./InsertAnything_tactile/scripts/python.sh scripts/train_normal_angle.py \
  --data "$INSERT_DATA_TRAIN" --output "$INSERT_ANGLE_MODEL" \
  --epochs 80 --batch_size 64 --seed 0
```

网络使用共享 CNN，每侧 128 维特征，拼接一个归一化开度后，经 256→128→1 MLP 输出有符号角度。图像缩放为 120×160，损失为按角度范围归一化的 MSE。按物体和角度区间分层，以独立姿态做约 70/15/15 划分；训练保存 splits、校准和标签约定。

独立测试集每物体 20 组，使用与训练不相交的种子区间，并保持角度/标定约定：

```bash
export INSERT_DATA_TEST="$INSERT_REPO/InsertAnything_tactile/outputs/normal_fresh_$(date +%Y%m%d_%H%M%S)"
./InsertAnything_tactile/scripts/python.sh scripts/collect_normal_angle_data.py \
  --task all --samples_per_task 20 --angle_range_deg -15 15 --angle_bins 5 \
  --offset_xy_mm 2 2 --depth_range_mm 0.2 1.0 --seed 181000 --headless \
  --output "$INSERT_DATA_TEST"
./InsertAnything_tactile/scripts/python.sh scripts/evaluate_normal_angle.py \
  --checkpoint "$INSERT_ANGLE_MODEL/best.pt" --data "$INSERT_DATA_TEST" \
  --output "outputs/normal_fresh_eval_$(date +%Y%m%d_%H%M%S)" --device cpu
./InsertAnything_tactile/scripts/python.sh scripts/predict_normal_angle.py \
  --checkpoint "$INSERT_ANGLE_MODEL/best.pt" \
  --h5 "$INSERT_DATA_TEST/LHole-III/samples.h5" --row 0 --show_label
```

评估会检查新数据种子与训练集是否冲突，输出 MAE、分位数、预测 CSV/NPZ 和图。数据索引及训练 manifest 保存绝对路径；移动数据后需要核对索引路径，不能只复制一个 `collection.json`。

## 动态角度校正与触觉 PPO

静态数据与动态插入可能存在接触域差异。动态采集环境本身需要一个兼容的角度网络，先显式传入已训练的静态 v4 模型进行引导采集，再从同一模型适配。触觉 PPO 随后使用适配后的模型：

```bash
export INSERT_DYNAMIC_DATA="$INSERT_REPO/InsertAnything_tactile/outputs/dynamic_$(date +%Y%m%d_%H%M%S)"
./InsertAnything_tactile/scripts/python.sh scripts/collect_dynamic_angle.py \
  --task InsertAnythingTactile6D-LHole-III-Direct-v0 \
  --angle_checkpoint "$INSERT_ANGLE_MODEL/best.pt" \
  --num_envs 4 --grasp_batches 20 --initial_tilt_deg 10 --seed 210000 --headless \
  --output "$INSERT_DYNAMIC_DATA/LHole-III"
export INSERT_DYNAMIC_MODEL="$INSERT_REPO/InsertAnything_tactile/outputs/dynamic_model_$(date +%Y%m%d_%H%M%S)"
./InsertAnything_tactile/scripts/python.sh scripts/adapt_dynamic_angle.py \
  --data "$INSERT_DYNAMIC_DATA" --checkpoint "$INSERT_ANGLE_MODEL/best.pt" \
  --output "$INSERT_DYNAMIC_MODEL" --epochs 60 --device cuda:0
```

触觉 PPO：

```bash
./InsertAnything_tactile/scripts/python.sh scripts/train_tactile_rl.py \
  --task InsertAnythingTactile6D-LHole-III-Direct-v0 \
  --angle_checkpoint "$INSERT_DYNAMIC_MODEL/best.pt" \
  --num_envs 4 --max_iterations 1000 --initial_tilt_deg 10 --seed 0 --headless
```

图像渲染和 Taxim 的开销与无触觉任务不同，所以这里从 4 环境开始验证，不能直接照搬 2048 环境。该历史策略曾出现均值动作退化，运行成功不代表学习成功；详见[触觉诊断](runbooks/README_insertanything_tactile_failure_20261013.md)。

评估需同时保留 PPO 训练配置和它所绑定的角度模型；参考：

```bash
export INSERT_TACTILE_POLICY=/absolute/path/to/tactile/training/run/nn/policy.pth
./InsertAnything_tactile/scripts/python.sh scripts/eval_tactile_rl.py \
  --policy_checkpoint "$INSERT_TACTILE_POLICY" \
  --angle_checkpoint "$INSERT_DYNAMIC_MODEL/best.pt" \
  --num_envs 16 --seeds 1001 1002 1003 1004 --headless \
  --output "outputs/tactile_eval_$(date +%Y%m%d_%H%M%S)"
```

如果转移到另一台机器，先检查保存配置中的角度模型绝对路径，通过 `--angle_checkpoint` 指定同一权重的新路径；程序在仿真启动前检查 SHA-256 必须与训练契约一致。使用 `--help` 查看选项，不可静默换用另一个估计器。

## 常见问题

- **`--num_envs: command not found`**：多行命令的反斜杠必须是行末最后一个字符，其后不能写注释或空格；也可合并为一行。
- **找不到 Isaac/TacEx 模块**：先确认 `INSERTANYTHING_PYTHON` 指向完整环境；CPU 测试依赖不能替代仿真依赖。
- **输出已存在**：采集、Angle 训练/评估和匹配队列采用独立目录保护已有结果，换新名称。
- **源文件哈希变化**：匹配队列要求两组使用同一代码；先停止队列再修改源码。更新 Git 仓库时也不要覆盖运行中使用的文件。
- **TCP 持续下压却停住、Fz 很小**：检查实际接触几何、摩擦、轴线/XY 误差和控制刚度。原接触观测不包括全部切向摩擦，小合力也可能来自抵消。
- **确定性/随机成绩差很多**：分别报告两种模式，并检查动作分布与接触停滞；不能仅凭这一现象断定奖励公式有 bug。

历史路径、图像和诊断原始产物保存在工作站，未上传到源码仓库。历史手册作为来源记录保留，当前可运行入口以本页及对应脚本参数为准。
