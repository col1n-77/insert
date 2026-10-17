# InsertAnything：六自由度插入、轴线奖励与视触觉扩展

基于 [mzhsoul/InsertAnything](https://github.com/mzhsoul/InsertAnything) 的研究代码，包含原版插入任务、无触觉角度奖励对照，以及双侧 GelSight Mini 视触觉扩展。更新至 **2026-10-17**。上游说明见 [README.upstream.md](README.upstream.md)，许可证见 [LICENSE](LICENSE)。

当前重点是验证：**在相同观测、初始化、控制和 PPO 配置下，单独增加轴线对准奖励，是否能提高插入成功率。** 最新代码加入了两组顺序训练、配置核对、冻结权重评估和结果汇总，并恢复原版的自适应 Actor 学习率。该轮重训在本发布快照时仍在进行，尚无完整对照结论。

## 导航

- [安装与完整命令](docs/WORKFLOWS.md)：环境检查、训练、评估、录像、触觉采集和角度模型。
- [奖励对照实验说明](docs/MATCHED_REWARD_STUDY.md)：共同参数、奖励公式、预算、队列管理与比较方法。
- [触觉模块](InsertAnything_tactile/README.md) · [无触觉 Angle 模块](InsertAnything_angle/README.md)
- [历史实验与诊断](docs/runbooks/README.md) · [修改记录](docs/CHANGES.md) · [发布验证](docs/VALIDATION.md)

## 1. 四类任务分别做什么

| Gym 前缀 | 传感器与输入 | 用途 |
| --- | --- | --- |
| `InsertAnything-*` | 原夹爪、原观测和力反馈 | 上游任务，以及显式开启在手偏角与六自由度后的基线训练 |
| `InsertAnythingAngle-*` | 无触觉；原观测追加实际在手角度 | 隔离验证轴线奖励；可选深度倾角门控 |
| `InsertAnythingTactile-*` | 双侧 GelSight Mini；触觉作为独立数据输出 | 采集、录像、校准；原策略不读取触觉图像 |
| `InsertAnythingTactile6D-*` | 双侧 RGB → 冻结角度估计器 → PPO 观测和奖励 | 触觉反馈的六自由度插入实验 |

六个几何/公差任务后缀为：

| 后缀 | 物体 |
| --- | --- |
| `CircleHole-I` | 圆形插销，Tol. I |
| `SquareHole-II` | 方形插销，Tol. II |
| `LHole-III` | L 形插销，Tol. III |
| `TriangleHole-IV` | 三角形插销，Tol. IV |
| `HexagonHole-III` | 六边形插销，Tol. III |
| `HexagonHole-IV` | 六边形插销，Tol. IV |

完整 ID 示例：`InsertAnythingAngle-LHole-III-Direct-v0`。匹配奖励队列目前专用于 LHole-III；单次训练入口可选择六个任务。

**默认配置与实验配置需要区分：**原版默认偏角 0°、锁定 roll/pitch；下面的基线训练命令显式设置 ±10° 和六自由度。Angle 默认已采用这两项设置。初始化时 TCP 竖直向下，插销绕夹持法向 **TCP +Y** 产生有符号在手偏角；执行阶段允许手腕倾转补偿。闭爪后的实际角度应以仿真状态为准，不直接把请求采样值当标签。

LHole-III 的原版 Actor/Critic 为 26/73 维，Angle 为 27/74 维；维度由任务配置计算。**权重不能跨观测结构直接混用。**

## 2. 代码与配置位置

```text
insert/
├── source/InsertAnything/             # 原任务、控制器、奖励、孔/插销 USD
├── scripts/                          # 原版安装、训练、检查和完整回合评估
├── InsertAnything_angle/
│   ├── source/insertanything_angle/   # 实际角度观测、轴线奖励、可选深度门控
│   ├── scripts/                      # 单次训练、匹配队列、冻结评估、汇总
│   └── tests/                        # 几何、门控、噪声和队列检查
├── InsertAnything_tactile/
│   ├── source/insertanything_tactile/ # 传感器、记录器及触觉六自由度环境
│   ├── source/tactile_pose/           # 数据、标注、CNN、单角度估计
│   ├── scripts/                      # 采集、训练、校准、评估和查看
│   └── tests/
└── docs/                             # 可移植运行说明、实验协议、历史手册
```

| 内容 | 文件 |
| --- | --- |
| 几何、公差、阶段奖励、接触力参数 | [insertanything_tasks_cfg.py](source/InsertAnything/InsertAnything/tasks/direct/insertanything/insertanything_tasks_cfg.py) |
| 初始化与控制参数 | [insertanything_env_cfg.py](source/InsertAnything/InsertAnything/tasks/direct/insertanything/insertanything_env_cfg.py) |
| 原奖励、复位及六自由度执行 | [insertanything_env.py](source/InsertAnything/InsertAnything/tasks/direct/insertanything/insertanything_env.py) |
| 原版 PPO 超参数 | [agents/](source/InsertAnything/InsertAnything/tasks/direct/insertanything/agents/) |
| Angle 配置与数学实现 | [env_cfg.py](InsertAnything_angle/source/insertanything_angle/env_cfg.py)、[math.py](InsertAnything_angle/source/insertanything_angle/math.py) |
| 传感器及分侧标定 | [config.py](InsertAnything_tactile/source/insertanything_tactile/config.py)、[tactile_calibration.py](InsertAnything_tactile/source/tactile_calibration.py) |
| 单角度 CNN | [normal_learning.py](InsertAnything_tactile/source/tactile_pose/normal_learning.py) |

## 3. 环境准备

上游验证组合为 Isaac Sim 5.0、Isaac Lab 2.2.1、Python 3.11。本工作站实际运行的版本如下；它是已验证环境记录，不是自动安装锁文件：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.16 |
| Isaac Sim / Isaac Lab | 5.1.0 / 2.3.0 |
| PyTorch / CUDA build | 2.7.0+cu128 |
| RL-Games | 1.6.1 |
| NumPy / Gymnasium | 1.26.0 / 1.2.0 |
| GPU | RTX 5090 D，约 32 GB |

先准备包含 Isaac Sim、Isaac Lab、Isaac Lab Tasks/RL、RL-Games、PyTorch、toml、wheel 的 Python 3.11 环境。参考 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 的安装说明。**本仓库安装脚本复用已有仿真环境，不会安装整套 Isaac Sim。**

```bash
git clone https://github.com/col1n-77/insert.git
cd insert
export INSERTANYTHING_BASE_PYTHON=/absolute/path/to/isaaclab-env/bin/python
bash scripts/setup_local_env.sh
./scripts/python.sh scripts/list_envs.py
```

也可跳过创建 `.venv-isaac51`，显式指定完整环境：

```bash
export INSERTANYTHING_PYTHON=/absolute/path/to/isaaclab-env/bin/python
./scripts/python.sh scripts/list_envs.py
```

根目录和扩展启动器均优先导入当前克隆的源码；扩展自动切换到自己的目录。下面命令均从仓库根目录调用，`outputs/`、`logs/` 则写在对应项目内。触觉需要额外的 TacEx、TacEx assets 与 Taxim 标定，见[完整说明](docs/WORKFLOWS.md#触觉的额外依赖)。

## 4. 当前推荐的奖励对照训练

以下命令会实际启动 **两组各 500 轮**的长训练，先 control，再 aligned。先用 `--plan_only` 检查计划；正式执行时去掉它：

```bash
./InsertAnything_angle/scripts/python.sh scripts/run_matched_reward_study.py \
  --study_name "LHole_III_matched_adaptive_$(date +%Y%m%d_%H%M%S)" \
  --num_envs 2048 --max_iterations 500 \
  --horizon 128 --seq_length 128 --minibatch_size 8192 \
  --save_frequency 100 --seed 42 \
  --learning_rate 0.0001 --lr_schedule adaptive --kl_threshold 0.008 \
  --initial_tilt_deg 10 --align_sigma_deg 5 \
  --eval_num_envs 128 --eval_seeds 1001 2002 \
  --plan_only
```

两组均为原观测＋真实在手角度、零新增角度噪声、±10° 初始在手偏角、六自由度控制、原力惩罚开启、深度倾角门控关闭。**唯一奖励差别是轴线权重 0/1**，不是拿旧 26 维模型与新 27 维模型直接比较。

Actor 从 `1e-4` 开始按 KL 自适应调整，`schedule_type=standard`、`kl_threshold=0.008`。当前 rl_games 的独立 Critic 对该配置仍使用 IdentityScheduler、保持 `1e-4`，与原版实际行为一致。两组规则相同，实际学习率可因 KL 不同而分化。

`2048 × 128 × 500 = 131,072,000` 次采样转移/组，计划 Actor 更新 64,000 次/组；AMP 跳步可使实际更新数略少。更多环境或轮数不直接保证收敛。2048 环境在本工作站验证过；4096 环境曾出现接触丢弃，不能仅按显存余量扩容。

队列会检查每组 29 项配置、比较两份完整配置、保存源码哈希；每组结束后评估最终及最佳训练回报权重。主要结论使用同轮数最终权重的成功率、轴线误差和确定性/随机模式表现。详见[实验协议](docs/MATCHED_REWARD_STUDY.md)。

## 5. 原版六自由度基线与单次 Angle 训练

原版任务显式开启偏角及六自由度，保留原奖励：

```bash
./scripts/python.sh scripts/rl_games/train.py \
  --task InsertAnything-LHole-III-Direct-v0 \
  --num_envs 128 --max_iterations 1000 --headless --seed 42 \
  env.initial_in_hand_tilt_deg=10.0 env.ctrl.lock_roll_pitch=false \
  agent.params.config.full_experiment_name="LHole_III_inhand10_6d_$(date +%Y%m%d_%H%M%S)"
```

只训练一个 Angle 模型：

```bash
./InsertAnything_angle/scripts/python.sh scripts/train_angle_rl.py \
  --task InsertAnythingAngle-LHole-III-Direct-v0 \
  --num_envs 2048 --max_iterations 500 --headless --seed 42 \
  --horizon 128 --seq_length 128 --minibatch_size 8192 \
  --initial_tilt_deg 10 --angle_noise_std_deg 0 \
  --align_weight 1 --align_sigma_deg 5 \
  --learning_rate 0.0001 --lr_schedule adaptive --kl_threshold 0.008 \
  --run_name "LHole_III_axis_adaptive_$(date +%Y%m%d_%H%M%S)"
```

单次入口为兼容历史命令，默认仍是 `constant`；新匹配队列默认 `adaptive`。推荐显式写出调度参数。若单独测试深度门控，追加 `--depth_axis_gate`，其默认容差 3°、衰减宽度 5°、下限 0.1；它不在上述严格对照中启用。

## 6. 奖励到底增加了什么

Angle 使用真实物体轴线与孔轴线夹角，而不是在手角度的绝对值：

$$\alpha=\arccos\!\left[\operatorname{clip}(\mathbf a_O^W\cdot\mathbf a_H^W,-1,1)\right]$$

$$r_t^{\rm new}=r_t^{\rm original}+w_{\rm axis}\left[\exp\left(-\left(\frac{\alpha_t}{\sigma}\right)^2\right)-1\right].$$

完全对准时新增项为 0，偏角增大时趋于负的权重值。原 XY、yaw、深度、动作、动作变化、阶段奖励和力惩罚保留。可选门控仅将连续深度项乘以倾角函数，不会再额外添加一份深度奖励，原阶段成功判定也不被门控替换。

触觉版先通过 TCP 世界姿态和预测在手角度恢复物体世界轴线。其历史奖励/控制尺度还包含其他改动，因此触觉版与本轮无触觉 Angle 实验不能当成只差传感器的一组严格对照。

## 7. 评估、录像和逐步状态

评估需要真实权重及其训练配置；下面占位路径必须替换。Angle 会保存冻结权重、模式/种子、原奖励和新增奖励、轴线误差、TCP/物体位姿、动作与接触力等状态：

```bash
export INSERT_ANGLE_RUN=/absolute/path/to/angle/training/run
export INSERT_ANGLE_CHECKPOINT="$INSERT_ANGLE_RUN/nn/your_checkpoint.pth"
./InsertAnything_angle/scripts/python.sh scripts/eval_angle_policy.py \
  --checkpoint "$INSERT_ANGLE_CHECKPOINT" --training_run "$INSERT_ANGLE_RUN" \
  --num_envs 128 --seeds 1001 2002 --modes deterministic stochastic \
  --headless --output "outputs/manual_eval_$(date +%Y%m%d_%H%M%S)"
```

追加 `--video --video_env 0` 可录制所选环境。主要文件为 `report.json`、`evaluated_checkpoint.pth`、各模式/种子下的 `states.npz`、`diagnostics.csv`、`summary.json`；视频及成功帧仅在录像开启、对应条件满足时生成。具体字段、原版评估和双触觉录像见[运行说明](docs/WORKFLOWS.md#评估输出与动作模式)。

确定性模式执行策略均值；随机模式从策略分布采样。两个模式均保留标准动作裁剪。比较时固定初始化分布、评估种子、回合数和权重选择规则，不用不同奖励定义下的总回报直接判定优劣。

## 8. 双侧触觉与单角度预测

传感器为 **GelSight Mini + Taxim**，使用 PhysX 刚体胶垫和深度渲染。左右分别以实际相机到胶面几何校准，RGB 为各自相机原生视角；两侧图像无需逐像素相同。该模型不包含 UIPC/FEM 软胶形变、标记剪切或由图像标定的接触力。

最新静态数据流程固定插销世界轴线竖直，围绕 TCP +Y 改变夹爪姿态，随机接触位置和 **0.2～1.0 mm 几何压入目标**。写入实际状态并渲染后读取标签，只接收有效双侧接触。这里的 `grasp_id` 是独立接触姿态分组编号，不表示每条样本都重新执行物理抓取。

单输出模型：

```text
左 RGB ─┐
        ├─ 共享 CNN（每侧 128 维）─ 拼接 ─ 加入实测两指关节位移和 ─ MLP ─ θ
右 RGB ─┘
```

模型图像输入缩放为 120×160，输出 `15 * tanh(...)` 范围内的一个有符号角度（范围随数据配置）；只预测绕夹持法向的在手偏角，不输出通用三维姿态。原完整相对旋转网络作为历史入口保留。

HDF5 保存左右 RGB/深度、两指位置与开度、TCP/物体世界位姿、物体相对 TCP 完整位姿及单角度标签、时间/步编号、环境/物体/接触姿态编号、随机种子、接触有效性和采集状态。训练按独立姿态划分，并检查校准 ID、坐标约定与新测试种子。完整命令和字段见[运行说明](docs/WORKFLOWS.md#单角度数据采集与训练)。

## 9. 已有结果与已知限制

历史 LHole-III 评估使用每模式 256 个回合：

| 既有策略 | 确定性成功率 | 随机成功率 |
| --- | --- | --- |
| 原奖励、原 26 维观测 | 78.13% | 81.64% |
| 原奖励＋轴线项、27 维观测 | 40.23% | 82.81% |
| 轴线项＋深度门控、27 维观测 | 43.36% | 79.69% |

这些策略的环境数、minibatch、优化更新数、学习率和训练预算不同，**不能证明轴线奖励单独导致提升或退化**；本次新增匹配队列正是为消除这些混杂。原奖励策略也有持续下压但卡住的失败回合。已有审计及来源见[代码与轨迹复核](docs/runbooks/README_insertanything_depth_axis_gate.md)和[历史模型比较](docs/runbooks/README_insertanything_reward_comparison_20261014.md)。

接触力观测是经过处理的接触法向合力；相对孔壁的法向力会互相抵消，切向摩擦不包含在该信号内。因此小 Fz 不能单独说明没有摩擦阻力。几何压入图也不能直接当压力图使用。

静态单角度模型曾在 120 个独立样本上达到约 0.751° MAE；这不等于动态插入时同样准确，触觉 RL 的历史训练仍有策略退化问题。当前仓库不宣称触觉策略已优于无触觉策略。

## 10. 测试、资产与可复现范围

已有 Python 环境可执行 CPU 检查：

```bash
export INSERTANYTHING_PYTHON=/absolute/path/to/python
bash scripts/test_cpu.sh -q
python3 InsertAnything_tactile/scripts/verify_isolation.py
```

独立 CPU 环境的依赖安装见[验证说明](docs/VALIDATION.md)。测试不会启动 Isaac Sim，不能替代 GPU 接触、触觉渲染或完整回合评估。GitHub Actions 运行同一测试入口。

仓库包含上游孔/插销 USD，以及创建本机触觉 USD 覆盖层的代码。**不包含实验权重、数据集、日志、视频、虚拟环境和外部 TacEx 资产。** 上游可选权重见 [checkpoint README](source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/README.md)；新增模型需自行训练或另外获得匹配产物。评估时一起保留权重、`params/env.yaml`、`params/agent.yaml` 和扩展的 `run.json`。

历史手册是原工作站记录，绝对路径与结果图只在原机器可用；新机器优先使用本 README 和 [WORKFLOWS.md](docs/WORKFLOWS.md)。上游 BSD-3-Clause 声明保留；Isaac Sim、TacEx、Taxim 与各自资产遵循其原有许可证。
