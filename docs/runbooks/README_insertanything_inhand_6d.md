# InsertAnything 原版：在手偏角 ±10°、六自由度动作与重新训练

更新：2026-10-13。统一索引：[README.md](README.md)。

项目：`/data/nju/Lincao/InsertAnything`。本页使用原版 Franka 和六个原任务，没有添加视触觉传感器。原有触觉扩展的命令仍见 [触觉六自由度 RL](README_insertanything_tactile_rl_6d.md)。

后续更新：触觉 `InsertAnythingTactile6D-*` 也已同步为竖直夹爪、插销绕 TCP Y 轴 ±10°，与本页新实验的姿态初始化一致；传感器抓持定位与各自控制/奖励参数仍保留。详见上面的触觉 RL 手册。

## 1. 本次实验的定义

按用户选择，初始化改变的是**插销在夹爪内的姿态**。夹爪仍使用原任务的初始朝向；在原来的插销放置位置上，以 TCP 为支点、TCP +Y（夹持法向）为轴，均匀采样 `theta ~ U(-10°, +10°)`。这沿用前面单角度实验的偏转轴，没有另外引入第二个随机角度。

设 `R_WT` 为 TCP 世界姿态，初始插销姿态/位置为 `R_WO, p_WO`，则新的放置为：

```text
Delta_R_W = R_WT · Ry(theta) · R_WT^T
R_WO_new  = Delta_R_W · R_WO
p_WO_new  = p_WT + Delta_R_W · (p_WO - p_WT)
```

因此旋转围绕现有抓持点发生，而不是围绕插销 USD 的底部原点。接着执行原来的物理闭爪过程；闭爪后的实际偏角可能因接触略有变化，不能把采样角当成实际测量值。

控制仍输出 6 个值 `[dx, dy, dz, rx, ry, rz]`。后三个是**世界坐标系的增量旋转向量**，按原来的 `rot_action_threshold` 缩放后转成四元数，左乘当前 TCP 姿态。关闭锁定后，不再强制将目标姿态投影到 `roll=π, pitch=0`。动作 EMA、增益、步长缩放、位置限制、物理参数均保持原实现。

两项新配置都位于通用环境配置中：

| 配置 | 本次训练取值 | 默认值 |
| --- | --- | --- |
| `env.initial_in_hand_tilt_deg` | `10.0` | `0.0`，原始抓持 |
| `env.ctrl.lock_roll_pitch` | `false` | `true`，原始控制 |

默认值保留了旧命令的行为；运行本页命令才启用在手偏角与六自由度控制。六个任务的资产、任务 ID、奖励、成功条件、观测字段及 PPO 网络配置没有修改。原有力反馈按各任务设置保留，未添加 GelSight 等图像传感器。

## 2. 从零训练 L 形任务

使用现有 Python 3.11 / Isaac Sim 5.1 本机环境，安装和版本检查见 [原版环境文档](README_insertanything.md)。这条命令不加载旧权重；每次创建独立的日志目录：

```bash
cd /data/nju/Lincao/InsertAnything
run_name="LHole_III_inhand10_6d_$(date +%Y%m%d_%H%M%S)"
./scripts/python.sh scripts/rl_games/train.py \
  --task InsertAnything-LHole-III-Direct-v0 \
  --num_envs 128 --headless --seed 42 \
  --max_iterations 1000 \
  env.initial_in_hand_tilt_deg=10.0 \
  env.ctrl.lock_roll_pitch=false \
  agent.params.config.full_experiment_name="$run_name"
```

输出目录：`/data/nju/Lincao/InsertAnything/logs/rl_games/InsertAnything/<run_name>/`。

- `params/env.yaml`：保存初始偏角和控制开关等环境设置。
- `params/agent.yaml`：保存本次 PPO 配置。
- `nn/`：新权重，正常训练后的最佳权重通常为 `InsertAnything.pth`，另有 `last_*.pth`。
- `summaries/`：TensorBoard 训练曲线。

`1000` 是训练迭代上限，不代表保证收敛。2026-10-12 的代码验证只运行少量 PPO 迭代；用户随后启动了完整训练，2026-10-13 的中途权重评估见第 8 节。

查看曲线：

```bash
cd /data/nju/Lincao/InsertAnything
./scripts/python.sh -m tensorboard.main --logdir logs/rl_games/InsertAnything --port 6006
```

浏览器访问 `http://localhost:6006`。

## 3. 评估新的策略

将下面的权重路径替换为本次训练产出的文件。**评估也必须传入这两个配置项**；当前原版 `play.py` 不会从 checkpoint 自动恢复它们。

```bash
cd /data/nju/Lincao/InsertAnything
./scripts/python.sh scripts/rl_games/play.py \
  --task InsertAnything-LHole-III-Direct-v0 \
  --num_envs 128 --headless --seed 43 --max_episodes 1 \
  --checkpoint /absolute/path/to/new_run/nn/InsertAnything.pth \
  env.initial_in_hand_tilt_deg=10.0 \
  env.ctrl.lock_roll_pitch=false
```

`--max_episodes 1` 完成一批并行评估。Hydra 配置覆盖中的角度请写成 `10.0`，因为本机配置解析严格区分浮点数与整数。L 形任务的 policy/critic 维度仍为 26/73，动作仍为 6；旧权重在维度上可加载，但旧策略没有在这些初始偏角和开放 roll/pitch 的执行方式下训练，不能沿用此前的成功率。

这次按要求保留原奖励和成功判定。原策略观测没有直接获得新增的在手角度真值；L 形任务可以使用原有力反馈和循环网络状态。原成功条件也没有新增轴线夹角门限，因此最终要判断倾斜插入的效果，还应检查实际轴线与插入轨迹。

## 4. 其余任务

上面的训练、评估命令替换 `--task` 即可，并为每个物体使用不同的 `full_experiment_name`：

| 任务 | 原任务 ID |
| --- | --- |
| 圆孔 I | `InsertAnything-CircleHole-I-Direct-v0` |
| 方孔 II | `InsertAnything-SquareHole-II-Direct-v0` |
| L 孔 III | `InsertAnything-LHole-III-Direct-v0` |
| 三角孔 IV | `InsertAnything-TriangleHole-IV-Direct-v0` |
| 六边形孔 III | `InsertAnything-HexagonHole-III-Direct-v0` |
| 六边形孔 IV | `InsertAnything-HexagonHole-IV-Direct-v0` |

各任务继续使用原本注册的 PPO 配置。不要用不同物体的权重混合评估。

## 5. 有限步物理检查

脚本会做三次独立重置，记录采样偏角、物理闭爪后的实际偏角和手指关节位置；分别检查三轴目标旋转和实际机械臂运动，再检查零动作数值、观测维度和任务配置是否保持不变。它不评估策略成功率。

```bash
cd /data/nju/Lincao/InsertAnything
./scripts/python.sh scripts/check_inhand_6d.py \
  --task InsertAnything-LHole-III-Direct-v0 \
  --num_envs 4 --headless --seed 71 --tilt_deg 10 \
  --report outputs/inhand_6d_check.json
```

加 `--legacy` 可以验证旧的零偏角、锁定 roll/pitch 默认行为。每次指定不同 `--report` 路径即可保留之前的检查结果。

## 6. 修改位置与历史记录

- 通用配置：`/data/nju/Lincao/InsertAnything/source/InsertAnything/InsertAnything/tasks/direct/insertanything/insertanything_env_cfg.py`。
- 初始插销偏转和动作执行：同目录 `insertanything_env.py`。
- 检查入口：`/data/nju/Lincao/InsertAnything/scripts/check_inhand_6d.py`。
- 本次验证证据：`/data/nju/Lincao/InsertAnything/outputs/inhand_6d_validation_20261012/`。

此前触觉扩展的隔离校验清单记录的是修改原项目之前的源文件哈希。由于本次用户明确要求修改原项目，其旧清单会指出这两个源文件不同；本次没有重写历史清单，也没有修改触觉项目的代码、数据或模型。

## 7. 2026-10-12 实测验证

六个任务各做 4 环境 × 3 次独立抓取（共 72 次）；三轴控制目标、实际机械臂响应、零动作和数值有限性检查均通过。旧默认模式另做 12 次抓取，也通过。

| 任务 | 闭爪后偏角相对采样值的平均变化 | 实际偏角范围 |
| --- | --- | --- |
| CircleHole-I | 0.0023° | -7.670° ～ 9.873° |
| SquareHole-II | 0.0013° | -7.671° ～ 9.872° |
| LHole-III | 0.1788° | -10.582° ～ 9.469° |
| TriangleHole-IV | 0.0118° | -7.660° ～ 9.885° |
| HexagonHole-III | 0.0038° | -9.589° ～ 9.478° |
| HexagonHole-IV | 0.0031° | -7.669° ～ 9.872° |

上述范围是少量样本的实际测量值，不是重新定义采样范围。L 形样本闭爪后最大约 −10.58°，体现了物理接触引起的偏角变化；没有把物体焊接到夹爪或每步强制重写姿态。

L 形任务另外完成 **4 环境、3 轮 PPO**（seed 71），保存了新策略。检查确认 checkpoint 的 epoch 为 3、优化器确实进行了更新、所有参数与优化器 tensor 均为有限值。使用 seed 72 重新加载，完成一批 4 回合评估，成功率为 **0/4**。这是短训练的流程检查结果，还没有得到可用的倾斜插入策略；需要执行第 2 节的完整训练后重新评估。

短训练权重：

```text
/data/nju/Lincao/InsertAnything/logs/rl_games/InsertAnything/smoke_inhand10_6d_20261012/nn/last_InsertAnything_ep_3_rew__53.47701_.pth
```

可复现的短训练命令（仍使用原任务的回合时长和 PPO rollout 参数）：

```bash
cd /data/nju/Lincao/InsertAnything
./scripts/python.sh scripts/rl_games/train.py \
  --task InsertAnything-LHole-III-Direct-v0 \
  --num_envs 4 --headless --seed 71 --max_iterations 3 \
  env.initial_in_hand_tilt_deg=10.0 \
  env.ctrl.lock_roll_pitch=false \
  agent.params.config.full_experiment_name="smoke_inhand10_6d_$(date +%Y%m%d_%H%M%S)"
```

证据目录中的 `physical_summary.json` 汇总六任务物理检查，`source_scope_check.json` 记录原任务资产、奖励、观测及其他源文件的保留检查，`ppo_checkpoint_check.json` 记录权重检查，`ppo_reload_check.json` 记录短策略回放结果。首次使用整数 `10` 的配置解析错误保留在 `ppo_train.log`；实际成功训练日志为 `ppo_train_float.log`。


## 8. 2026-10-13：训练中的第 846 轮策略评估

用户运行目录：`/data/nju/Lincao/InsertAnything/logs/rl_games/InsertAnything/LHole_III_inhand10_6d_20261012_233230`。
从当时最新的最佳奖励权重 `nn/InsertAnything.pth` 复制独立快照，读取到 epoch=846、frame=13,860,864。评估没有停止或更新训练进程。

快照：`/data/nju/Lincao/InsertAnything/outputs/inhand_rl_eval_running_20261013/evaluated_checkpoint.pth`。SHA256：`9893935e8dc32d0f122ee6474e14e89f5189cefd862cf636b124d177907a81ef`。

使用新脚本 `scripts/eval_inhand_policy.py`，自动恢复该次 `params/env.yaml` 和 `params/agent.yaml`，保留在手偏角 ±10°、开放 roll/pitch、原夹爪、原奖励和原观测。测试 seed=1001，128 环境；分别测试原 player 默认的随机动作与确定性均值动作。原任务设置 20 秒，沿用原结束逻辑，实际记录 299 个控制步、19.933 秒；未缩短回合。

| 动作模式 | 完整回合 | 原版成功判定 | 加真实轴线/深度检查 | 回合结束仍满足严格条件 |
| --- | ---: | ---: | ---: | ---: |
| 随机动作（原 player 默认） | 128 | 108/128 (84.38%) | 108/128 (84.38%) | 108/128 |
| 确定性均值动作 | 128 | 109/128 (85.16%) | 109/128 (85.16%) | 109/128 |

成功按整个回合至少一次满足条件统计，并另报末帧成功。原条件为 XY <2.5 mm、Z 深度误差 <1 mm、对称性 yaw 误差 <0.1 rad。额外严格条件为真实物体/孔轴夹角 <3°、Z 深度误差不低于 −2 mm。没有为无触觉版本添加触觉有效性条件，也没有修改奖励或终止逻辑。

随机动作平均首次严格成功时间 8.66 秒；全部回合末帧平均真实轴线误差 0.979°。确定性模式对应末帧平均误差 0.887°。这是一个新种子的阶段性评估，不代表已经完成多种子泛化验证。

复现这份固定权重的评估：

```bash
cd /data/nju/Lincao/InsertAnything
./scripts/python.sh scripts/eval_inhand_policy.py \
  --checkpoint outputs/inhand_rl_eval_running_20261013/evaluated_checkpoint.pth \
  --training_run logs/rl_games/InsertAnything/LHole_III_inhand10_6d_20261012_233230 \
  --task InsertAnything-LHole-III-Direct-v0 \
  --num_envs 128 --seeds 1001 --modes stochastic deterministic --headless \
  --output "outputs/inhand_eval_repeat_$(date +%Y%m%d_%H%M%S)"
```

本脚本从训练配置恢复姿态设置，因此不再接收 Hydra 的 `env.*` 覆盖。原来的 `scripts/rl_games/play.py` 仍需使用第 3 节的两个姿态参数。

结果目录：`/data/nju/Lincao/InsertAnything/outputs/inhand_rl_eval_running_20261013`。
`report.json` 保存逐回合成功及配置；`summary.json` 保存汇总；`evaluation_diagnostics.png` 是两模式曲线；各模式子目录中的 `diagnostics.csv` 和 `states.npz` 保存逐步动作、关节位置/速度、TCP/物体/孔位姿及原任务接触力。所有末帧数据在自动复位前读取，四元数 wxyz，世界位置单位米。`raw_actions` 为策略限幅后、EMA 前动作；`policy_obs_before_action` 是相应动作之前的输入。


## 9. 2026-10-13：第 846 轮策略的成功插入视频

已使用第 8 节的固定权重，在原版无触觉 `InsertAnything-LHole-III-Direct-v0` 中重新执行并录制一个完整成功回合。训练配置自动恢复，仍为竖直夹爪、插销绕夹持法向随机 ±10°、开放 roll/pitch；只调整观看相机。单环境、确定性动作、seed=1001。这是可视化样例，不计入第 8 节的 128 回合统计。

[播放成功视频](/data/nju/Lincao/InsertAnything/outputs/inhand_success_video_epoch846_20261013/deterministic_seed_1001/scene.mp4)。视频为真实策略执行时渲染，1280×720、15 FPS、299 帧、19.933 秒；每个控制步采样一次，在自动复位前保存末帧。

本回合采样初始偏角为 +4.633°；7.733 秒首次同时满足原成功条件和真实轴线/深度检查，之后保持成功直到结束。末帧物体/孔轴夹角 0.305°、XY 误差 0.067 mm、深度误差 +0.043 mm。目录中保留 `report.json`、`video_audit.json`、脚本快照、运行日志及同步 `diagnostics.csv` / `states.npz`。

重新录制（输出目录必须尚不存在；无需停止已有训练）：

```bash
cd /data/nju/Lincao/InsertAnything
./scripts/python.sh scripts/eval_inhand_policy.py \
  --checkpoint outputs/inhand_rl_eval_running_20261013/evaluated_checkpoint.pth \
  --training_run logs/rl_games/InsertAnything/LHole_III_inhand10_6d_20261012_233230 \
  --task InsertAnything-LHole-III-Direct-v0 \
  --num_envs 1 --seeds 1001 --modes deterministic --headless --video \
  --output "outputs/inhand_video_$(date +%Y%m%d_%H%M%S)"
```

`--video` 会自动启用渲染；可通过 `--camera_eye x y z` / `--camera_lookat x y z` 调整机位。多环境时，`--video_env` 选择被录制的环境（默认 0），成功率仍分别统计所有环境。录制本身不筛选成功回合，须以对应报告中的成功判定为准。

在桌面打开已录制视频：

```bash
xdg-open /data/nju/Lincao/InsertAnything/outputs/inhand_success_video_epoch846_20261013/deterministic_seed_1001/scene.mp4
```


## 10. 2026-10-14：训练完成后的原奖励/轴线奖励对比

本 run 已完成第 1,000 轮。本次对最终权重及最佳第 846 轮分别使用 seeds 1001/2002、128 环境、两种动作模式评估，各权重 512 回合。最终权重随机严格成功 209/256（81.64%），确定性 200/256（78.13%）；最佳权重分别 212/256（82.81%）及 210/256（82.03%）。这是一组新评估，上述历史单种子结果继续保留。与无触觉 Angle 模型的完整比较、逐步数据和复现命令见[现有策略奖励对比](README_insertanything_reward_comparison_20261014.md)。
