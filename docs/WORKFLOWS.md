# 统一运行说明

以下命令从仓库根目录执行。原版和扩展的 `scripts/python.sh` 默认共用根目录 `.venv-isaac51`；也可通过 `INSERTANYTHING_PYTHON` 指定完整 Isaac Lab 环境的 Python 绝对路径。扩展脚本自动切换到各自目录，并优先导入本仓库原版源码。`INSERTANYTHING_ROOT` 可覆盖扩展所复用的原版根目录，默认就是本仓库。

## 环境准备

先安装 Python 3.11、Isaac Sim / Isaac Lab、Isaac Lab RL / Tasks、RL-Games、PyTorch、toml 和 wheel。

```bash
export INSERTANYTHING_BASE_PYTHON=/absolute/path/to/isaaclab-env/bin/python
bash scripts/setup_local_env.sh
./scripts/python.sh scripts/list_envs.py
```

安装脚本借用指定环境中的包，创建本地 `.venv-isaac51`，以 editable 模式安装原版包。已有环境与指定 Python 不匹配时会停止并保留环境；脚本不安装 Isaac Sim、Isaac Lab 或 TacEx。使用已有完整环境时可跳过创建：

```bash
export INSERTANYTHING_PYTHON=/absolute/path/to/isaaclab-env/bin/python
./scripts/python.sh scripts/list_envs.py
```

六个任务后缀：`CircleHole-I`、`SquareHole-II`、`LHole-III`、`TriangleHole-IV`、`HexagonHole-III`、`HexagonHole-IV`。完整 ID 为 `<系列>-<后缀>-Direct-v0`。

## 原版与在手偏角六自由度

有限检查：

```bash
./scripts/python.sh scripts/smoke_test.py \
  --task InsertAnything-LHole-III-Direct-v0 \
  --num_envs 4 --steps 100 --headless --report outputs/base_smoke.json

./scripts/python.sh scripts/check_inhand_6d.py \
  --num_envs 4 --headless --report outputs/inhand_6d.json
```

开启 ±10° 在手偏角和六自由度训练：

```bash
./scripts/python.sh scripts/rl_games/train.py \
  --task InsertAnything-LHole-III-Direct-v0 \
  --num_envs 128 --max_iterations 1000 --headless --seed 42 \
  env.initial_in_hand_tilt_deg=10.0 env.ctrl.lock_roll_pitch=false \
  agent.params.config.full_experiment_name=inhand10_6d
```

评估六自由度模型时，先填写真实训练目录和匹配的权重路径：

```bash
export RUN=/absolute/path/to/training/run
export CHECKPOINT=/absolute/path/to/training/run/nn/policy.pth
./scripts/python.sh scripts/eval_inhand_policy.py \
  --checkpoint "$CHECKPOINT" --training_run "$RUN" \
  --num_envs 128 --seeds 1001 2002 --modes deterministic stochastic \
  --headless --output outputs/inhand_eval
```

输出目录需要是新目录，脚本会冻结权重副本并记录完整回合。训练目录需要原始 `params/env.yaml` 和 `params/agent.yaml`。

## 无触觉角度观测与奖励

```bash
./InsertAnything_angle/scripts/python.sh scripts/train_angle_rl.py \
  --task InsertAnythingAngle-LHole-III-Direct-v0 \
  --num_envs 128 --max_iterations 1000 --headless --seed 42 \
  --angle_noise_std_deg 0 --initial_tilt_deg 10 \
  --align_weight 1 --align_sigma_deg 5 --run_name angle_axis_seed42
```

默认新增项为 `w * (exp(-(alpha/sigma)^2) - 1)`，其中 `alpha` 是真实插销/孔轴线夹角，对准时新增项为 0。设置 `--align_weight 0` 可保留同样观测并关闭新增奖励。

可选深度倾角门控：在上述训练命令中追加 `--depth_axis_gate --depth_gate_tolerance_deg 3 --depth_gate_sigma_deg 5 --depth_gate_floor 0.1`。门控只调整连续深度奖励，默认关闭；使用真实插销/孔轴线夹角，不是手腕倾角。

```bash
./InsertAnything_angle/scripts/python.sh scripts/eval_angle_policy.py \
  --checkpoint "$CHECKPOINT" --training_run "$RUN" \
  --num_envs 128 --seeds 1001 2002 --modes deterministic stochastic \
  --headless --output outputs/angle_eval
```

此处 `RUN` 和 `CHECKPOINT` 必须改为 Angle 训练目录与对应权重；保留其 `run.json` 和 `params/`。评估入口会恢复保存的奖励形式。两个动作模式分别取策略均值和从策略分布采样，训练/评估随机种子需区分。

Angle 的 `scripts/run_reward_study.py` 会实际依次启动同配置长训练和评估。先查看 `--help`，为新实验使用新的 `--study_name`；本次整理未运行该研究。

## 触觉采集、角度网络与 PPO

额外准备独立安装的 `tacex`、`tacex_assets`、GelSight Mini 网格和 Taxim 标定。数据处理工具需要 NumPy、SciPy、h5py、PyTorch；绘图/视频入口还需要 Matplotlib、imageio 等对应依赖。

首次创建触觉环境时，由已有 `prepare_robot_asset()` 生成本机 robot USD 覆盖层。重定位 TacEx 后需重新生成，见[资产说明](../InsertAnything_tactile/assets/README.md)。该实现使用 PhysX 刚体接触与 Taxim 光学，不含 UIPC 胶体形变或标记剪切。

```bash
./InsertAnything_tactile/scripts/python.sh scripts/check_configs.py --headless
```

下表入口位于 `InsertAnything_tactile/scripts/`，使用扩展的 `scripts/python.sh` 调用，先查看相应 `--help`：

| 脚本 | 用途 |
| --- | --- |
| `capture.py`、`record_eval.py`、`inspect_recording.py` | 同步双侧触觉和状态 |
| `validate_tactile_calibration.py` | 分侧光学校准 |
| `collect_pose_data.py`、`train_pose.py`、`evaluate_pose.py` | 相对旋转姿态数据和模型 |
| `collect_tcp_pose_data.py` | 六物体 TCP 约束采样 |
| `collect_normal_angle_data.py`、`train_normal_angle.py` | 单角度静态采集和网络训练 |
| `evaluate_normal_angle.py`、`predict_normal_angle.py` | 独立测试和推理 |
| `collect_dynamic_angle.py`、`adapt_dynamic_angle.py` | 动态角度采集和模型适配 |
| `train_tactile_rl.py`、`eval_tactile_rl.py` | 触觉六自由度 PPO |

触觉 PPO 需要先提供兼容的动态角度网络权重，权重未上传：

```bash
export ANGLE_CHECKPOINT=/absolute/path/to/dynamic-angle/best.pt
./InsertAnything_tactile/scripts/python.sh scripts/train_tactile_rl.py \
  --task InsertAnythingTactile6D-LHole-III-Direct-v0 \
  --angle_checkpoint "$ANGLE_CHECKPOINT" --num_envs 4 \
  --max_iterations 1000 --headless --enable_cameras
```

各代数据和模型存在不同 schema、坐标约定及标定。保持数据、角度估计器和 PPO 训练配置相匹配。模型默认路径保留原实验命名，实际运行时显式传入本机文件。

## CPU 验证和历史记录

```bash
export INSERTANYTHING_PYTHON=/absolute/path/to/python
bash scripts/test_cpu.sh -q
python3 InsertAnything_tactile/scripts/verify_isolation.py
```

CPU 测试不启动 Isaac Sim 或 GPU。完整性检查针对本次发布快照的相对路径与 SHA-256；后续主动修改源码后报告变化属于预期，它不检查外部 Isaac/TacEx 安装。

原工作站命令、结果、模型路径和实验限制见[历史手册](runbooks/README.md)。归档中的绝对路径、图片和结果链接指向原工作站，相关数据不在此仓库内。本次上传整理未启动新的长训练、采集或成功率评估。
