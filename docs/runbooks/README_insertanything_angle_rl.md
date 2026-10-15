# InsertAnything Angle：无触觉、无角度噪声的轴线惩罚对照

更新：2026-10-15。默认附加奖励版本中，每一步在原奖励上加入 `w*(exp(-(alpha/sigma)^2)-1)`，完全对准时新增项为 0；新增在手角度观测噪声为 0。原任务已有观测噪声、物理随机化与 ±10° 初始在手偏角继续保留。独立代码目录 `/data/nju/Lincao/InsertAnything_angle`，原版与触觉版代码、任务及既有权重保持原样。

2026-10-15 新增可选[深度奖励倾角门控](README_insertanything_depth_axis_gate.md)，需显式传 `--depth_axis_gate`；本页此前实验及训练命令仍描述门控关闭的附加奖励版本。旧模型评估会自动保持门控关闭。

## 1. 实验问题与版本边界

本实验验证：在修改后的原版 InsertAnything 中，使用“原有观测＋无附加噪声的实际在手角度”，原奖励加轴线对准奖励能否改善插入成功率和轴线对准。

不添加触觉传感器，也不加载触觉角度网络。保留原版的力观测、场景资产、质量/摩擦/增益随机化、原观测噪声、控制器、抓取复位、原奖励及原回合终止逻辑。新增注册六个独立任务；本次正式对照训练先用 `InsertAnythingAngle-LHole-III-Direct-v0`。

| 项目 | 设置 |
| --- | --- |
| 初始夹爪 | 原版竖直向下姿态 |
| 初始物体偏角 | 绕夹持法向 TCP +Y，采样均匀 ±10° |
| 角度观测 | 从实际物体/TCP 旋转计算，不添加角度观测噪声，`noise_std_deg=0` |
| Actor | 原 L 任务 26 维观测＋1 维角度，共 27 维 |
| Critic | 原 73 维状态＋1 维真实角度，共 74 维；仅训练使用 |
| 角度归一化 | 角度 / 15°；不是把角度裁剪到 ±15° |
| 动作 | 原 6 维动作，`lock_roll_pitch=false` |
| 新奖励 | 真值轴线误差的 `exp(-(alpha/sigma)^2)-1` 惩罚，逐控制步纯加法；原插入/阶段奖励保持原样 |
| 可选噪声开关 | 保留独立 generator 与缓存实现；当前两组均设为 0，不扰动新增角度输入 |

这里的 ±10° 是**复位时采样的偏角**，闭爪和物理接触后实际偏角可能略超出。策略读取每一步实际姿态计算的偏角，不能使用初始采样值冒充当前角度。此前 1° 观测噪声版本的 128 环境初检中，实际角度范围为约 −13.00°～9.91°，TCP 偏离竖直最大 0.050°；这是历史检查结果，不能当作本次零噪声新训练的结果。

## 2. 坐标与奖励定义

物体/孔插入轴均取各自局部 +Z。零偏角物体轴在向下的 TCP 中为 −Z。将实际物体轴转换到 TCP 后记为 `a_T`：

```text
theta_true = atan2(-a_T.x, -a_T.z)
theta_observed = theta_true  # angle_noise_std_deg = 0
actor_angle = theta_observed / (15 degrees)

alpha = atan2(norm(cross(a_object_world, a_hole_world)),
              dot(a_object_world, a_hole_world))
r_axis = exp(-(alpha / (5 degrees))**2) - 1
r_align = weight * r_axis
r_total = r_original + r_align
```

奖励中的 `alpha` 使用**仿真真实物体与孔轴线**，以隔离观测误差与奖励本身的效果；策略只得到当前实际在手单角度，不得到新增的真值轴线观测。此次结果不能直接等同于触觉网络参与奖励时的效果。

单角度表示假设物体主要绕 TCP Y 偏转。评估还记录离开 TCP XZ 平面的轴线残差 `off_axis_residual_deg`，以识别滑移/多轴旋转造成的模型不足。

默认 `weight=1, sigma=5°`：新增项在 0° 时为 **0**，5° 时约 **−0.6321**，10° 时约 **−0.9817**，有界于 `[-weight, 0]`。权重只乘一次。代码用 `torch.expm1` 稳定计算 `exp(x)-1`，并保持原奖励各项不变。

减 1 等于每一步减去常数 `weight`。在当前固定回合长度下，任意两条轨迹间的回报差保持不变；它改变奖励零点及 critic 的学习目标，不能单凭正负号推断更容易学会纠偏。大偏角时惩罚接近下限，对角度的敏感度依然较弱；`sigma=5°` 的边界没有因减 1 消失。

本版本保存 `axis_alignment.zero_at_alignment=true` 与 `axis_reward_form=gaussian_minus_one`。历史权重若没有此字段，评估脚本明确恢复旧的正值 Gaussian 形式并写入报告，避免静默混用两个奖励定义。

## 3. 同配置奖励消融

| 实验 | Actor 输入 | 噪声 | 原奖励 | 新增奖励 |
| --- | --- | --- | --- | --- |
| aligned | 原观测＋实际角度 | 0° | 保留 | `weight=1` |
| control | 原观测＋实际角度 | 0° | 保留 | `weight=0` |

两组均从头训练，seed=42，128 环境，horizon=128，minibatch=512，sequence=128，4 mini epochs，预算各 1,000 轮，即各 16,384,000 个环境转移。原策略观测维度不同，不直接加载为新模型。

保留原 PPO 网络及其余主要超参数；actor 与 central critic 的学习率均固定为 `1e-4`，防止此前触觉版自适应学习率持续升高。固定学习率也需要结合 KL/动作饱和指标观察，不能保证自动收敛。

主比较使用**相同最终轮数**的权重；另评估各自训练回报最佳权重作为补充。由于奖励定义不同，不比较两组总回报的绝对高低。只用一个训练 seed 的结果属于首轮验证，不代表跨训练种子的稳定统计结论。

## 4. 环境与短检查

复用 `/data/nju/Lincao/InsertAnything/.venv-isaac51`，本机为 Isaac Sim 5.1 / Isaac Lab 2.3 / Python 3.11。`scripts/python.sh` 自动设置新包的 PYTHONPATH，无需改原版安装。

```bash
cd /data/nju/Lincao/InsertAnything_angle
./scripts/python.sh scripts/check_angle_env.py --headless --num_envs 128 --steps 8 --output "outputs/contract_check_$(date +%Y%m%d_%H%M%S).json"
```

历史检查（1° 噪声配置）：纯 Torch 单元测试 7/7；128 环境物理检查 79/79。结果保留在 `outputs/contract_check_20261013.json`。当时 1,024 个观测噪声样本实测标准差 0.987°。检查覆盖原观测前缀精确一致、原全局 RNG 一致、原奖励保留、weight=0 精确消融、初始化、六任务配置隔离、局部复位与缓存。这些历史结果不表示零噪声新训练已完成或已证明奖励有效。

此前零噪声、正值奖励配置已完成 128 环境物理检查 **79/79**：1,024 个角度噪声值均严格为 0，actor 追加角度与实际在手偏角归一化值逐项相等。记录为 `outputs/contract_check_noise0_20261013.json`。

当前新惩罚公式通过纯 Torch **8/8** 检查及 128 环境物理检查 **90/90**，包括原奖励精确保留、权重为 0 时精确恢复原奖励、惩罚非正且范围正确、公式一致、每步相加及 1,024 个角度噪声值全部为 0。记录为 `outputs/contract_check_axis_penalty_20261013.json`。

PPO 冒烟训练仅检验训练和权重读取链路，不用于判断奖励是否有效。正式学习结论需要下面完整训练后的评估。

## 5. 训练命令

单独训练新增轴线奖励版本：

```bash
cd /data/nju/Lincao/InsertAnything_angle
./scripts/python.sh scripts/train_angle_rl.py \
  --task InsertAnythingAngle-LHole-III-Direct-v0 \
  --num_envs 128 --headless --seed 42 \
  --max_iterations 1000 \
  --initial_tilt_deg 10 \
  --angle_noise_std_deg 0 \
  --align_weight 1 --align_sigma_deg 5 \
  --learning_rate 0.0001 \
  --run_name "LHole_III_angle0_axis_penalty1_$(date +%Y%m%d_%H%M%S)"
```

仅将 `--align_weight 0` 并使用新 run_name，即为同角度观测的原奖励对照。

推荐完整对照入口，依次运行 aligned 训练/评估，再运行 control 训练/评估，最后生成比较报告：

```bash
cd /data/nju/Lincao/InsertAnything_angle
./scripts/python.sh scripts/run_reward_study.py \
  --study_name "LHole_III_angle0_axis_penalty_$(date +%Y%m%d_%H%M%S)" \
  --max_iterations 1000 --num_envs 128 --seed 42 \
  --angle_noise_std_deg 0 --eval_seeds 1001 2002
```

若本次已启动的对照仍在运行，不要重复启动同规模 GPU 作业。所有输出目录要求不存在，避免覆盖。脚本会核对源文件哈希，长实验途中修改参与训练的代码会阻止后续阶段混用版本。

## 6. 评估与查看

完整脚本默认对每个冻结权重评估两种模式（deterministic、stochastic），每种模式使用 seed=1001/2002、各 128 个环境，即每种模式 256 个完整回合。每回合按原终止规则记录 299 步、约 19.933 秒仿真时间。

手动评估时，把 CHECKPOINT 换成当前新版本的实际 `.pth` 路径；默认自动读取其训练目录中的完整 env/agent 配置和 run.json：

```bash
cd /data/nju/Lincao/InsertAnything_angle
./scripts/python.sh scripts/eval_angle_policy.py \
  --checkpoint CHECKPOINT \
  --num_envs 128 --headless --seeds 1001 2002 \
  --modes deterministic stochastic \
  --output "outputs/manual_eval_$(date +%Y%m%d_%H%M%S)"
```

评估默认恢复权重对应训练配置中的噪声值；本次零噪声权重会自动恢复为 0。需要显式指定时可加 `--noise_std_deg 0`，这是记录在报告中的评估覆盖项，不改变训练权重。视频可加 `--video --num_envs 1 --seeds 1001 --modes deterministic`；单环境视频不能替代 128 环境成功率评估。

评估指标：

- 原始成功率 native：沿用原任务判定。
- 严格成功率 strict：原判定通过，且真实轴线夹角 <3°、深度误差 ≥−2 mm。
- 回合末严格成功率：区分曾短暂达到成功与最终保持成功。
- 末端轴线夹角均值/中位数/P95、XY/深度误差、实际在手角、离轴残差、动作饱和比例。

所有状态在动作执行后、自动复位前保存。`policy_obs_before_action` 是对应动作执行前实际收到的输入；同一行中 post-action 的角度不是那个动作使用的旧角度。

每个训练 run 的目录为 `logs/rl_games/InsertAnythingAngle/<run_name>/`：

```text
run.json                 配置合同、运行状态、预算与权重路径
params/env.yaml          实际生效的环境配置
params/agent.yaml        实际生效的 PPO 配置
source_snapshot/         新增代码与原任务代码快照
source_manifest.json     源路径和 SHA256
nn/*.pth                 最佳/定期/最终权重
summaries/               TensorBoard
```

对照输出 `outputs/<study_name>/`：

```text
study.json               当前阶段、进程 PID、训练/评估路径
train_aligned.log        新增奖励版训练日志
train_control.log        对照训练日志（轮到该阶段时生成）
eval_*/report.json       冻结权重 SHA256、完整评估汇总
eval_*/<mode>_seed_*/    diagnostics.csv、states.npz、reset.json、summary.json
comparison.json          两组全部完成后生成的成功率/角度比较
```

尚未生成 `comparison.json` 或 `study.json` 不是 completed 时，不应宣称已验证奖励有效。训练耗时依负载变化；此前空闲阶段约 33 秒/轮，完整两组通常需约 18 小时及评估时间，历史并发负载下更久。

## 7. 原 128 环境对照队列的实际状态

此前启动的对照为 `LHole_III_angle0_axis_penalty_20261013`：计划两组从头训练，角度噪声均为 0；aligned 使用原奖励＋用户指定的惩罚项，control 权重为 0。

2026-10-13T13:12:02.455108+08:00 启动，原总控 PID `3907120`，计划各 128 环境、1,000 轮、seed=42。2026-10-14 检查确认该队列已退出；aligned 日志最后完成第 196 轮，随后出现 `carb.tasking/Mutex.cpp` 的 `Recursion not allowed` 断言。原奖励 control 未生成，自动对照未完成。历史 `study.json` / `run.json` 仍留有 running 字段，不能据此判断进程还在运行；现有记录与权重完整保留。用户随后启动并完成的 2,048 环境、100 轮实验及新评估见第 10 节。

此前两个实验均已停止并保留，不续训、不混入当前对照：

- `LHole_III_angle1_reward_ablation_20261013`：1° 噪声、正值 Gaussian，完成第 8 轮后按用户要求停止。
- `LHole_III_angle0_reward_ablation_20261013`：0° 噪声、正值 Gaussian，完成第 13 轮后按用户新公式停止；最佳权重 `nn/InsertAnythingAngle.pth` 及日志保留。

- 总状态：`/data/nju/Lincao/InsertAnything_angle/outputs/LHole_III_angle0_axis_penalty_20261013/study.json`
- 总控日志：`/data/nju/Lincao/InsertAnything_angle/LHole_III_angle0_axis_penalty_20261013.launcher.log`
- 新奖励训练：`logs/rl_games/InsertAnythingAngle/LHole_III_angle0_axis_penalty_20261013_aligned`
- 原奖励对照：`logs/rl_games/InsertAnythingAngle/LHole_III_angle0_axis_penalty_20261013_control`（队列轮到时创建）
- 最终比较：`outputs/LHole_III_angle0_axis_penalty_20261013/comparison.json`（两组训练/评估全部完成后生成）

查看当前状态和第一组训练进度：

```bash
cd /data/nju/Lincao/InsertAnything_angle
cat outputs/LHole_III_angle0_axis_penalty_20261013/study.json
tail -n 30 outputs/LHole_III_angle0_axis_penalty_20261013/train_aligned.log
```

此前 1° 噪声版本正式启动前已经完成两轮 128 环境 PPO 保存测试，随后在 8 环境分别运行 deterministic/stochastic 完整 299 步，两个模式各完成 8 回合；配置恢复、非有限值检查、动作和状态保存、指标汇总全部通过。两轮模型尚未学会插入，两个模式都是 0/8；这只是链路验证，不代表正式训练结果。记录在 `logs/rl_games/InsertAnythingAngle/contract_smoke_20261013` 和 `outputs/eval_contract_smoke_20261013`。

两轮冒烟训练总计 256 个控制步，未满一个 299 步回合，所以 checkpoint 名称中的 `rew_-inf` 是没有已完成回合回报的占位元数据，不是观测或网络参数包含无穷值。完整重载评估已通过有限值检查。

正式启动器使用 IsaacAlgoObserver，将新增对准奖励、真值轴线误差、离轴残差和严格成功比例记入 TensorBoard。这些训练标量是 rollout 末步快照，完整回合成功率以评估报告为准。

## 8. 代码入口

- `source/insertanything_angle/env_cfg.py`：初始偏角、噪声、奖励参数及六任务配置。
- `source/insertanything_angle/math.py`：角度几何、Gaussian 奖励、独立噪声缓存。
- `source/insertanything_angle/env.py`：追加观测和奖励，原环境继承。
- `scripts/train_angle_rl.py`：从头训练，固定学习率、源代码/配置快照。
- `scripts/run_reward_study.py`：完整两组训练与评估队列。
- `scripts/eval_angle_policy.py`：冻结权重、完整回合与可选视频。
- `scripts/summarize_reward_study.py`：比较报告；单组或结果不全不会伪造差值。

六个注册任务分别为 `InsertAnythingAngle-{CircleHole-I,SquareHole-II,LHole-III,TriangleHole-IV,HexagonHole-III,HexagonHole-IV}-Direct-v0`。已核对六者配置继承；当前实测物理和正式训练针对 LHole-III。


## 9. RTX 5090 D 加速配置与显存实测（2026-10-13）

以下只适用于本独立无触觉 Angle 版本。保留原网络、128 步 rollout、128 步 LSTM 序列、原仿真步长/控制频率及奖励；调整并行环境和 PPO minibatch。已完成每组两轮短测试，验证可启动、更新和保存权重，尚未证明这些批量设置的收敛效果。

| 并行环境 | minibatch | 总吞吐（环境转移/秒） | 训练进程显存 |
| --- | --- | --- | --- |
| 128 | 512 | 约 545，原正式训练近期中位数 | 约 5.5 GiB，运行时观测 |
| 512 | 512 | 1,775 | 短测观测峰值 5.8 GiB |
| 1,024 | 512 | 2,756 | 短测观测峰值 6.1 GiB |
| **2,048** | **8,192** | **7,233** | **短测观测峰值 7.8 GiB** |
| 2,048 | 65,536 | 7,598 | 短测观测峰值 12.3 GiB |

本机 GPU 为 RTX 5090 D，报告总显存 32,607 MiB。测试时临时暂停既有训练，暂停进程仍占约 5.5 GiB；表格只列被测训练进程的用量，不把另一进程和桌面的占用算进去。所有短测结束后原正式训练当时已恢复；该 128 环境队列后来异常退出，当前状态见第 7 节。

**4,096 环境暂不采用。** 该短测的现有 PhysX `collisionStackSize` 缓冲区溢出，明确出现 `Contacts have been dropped`。即使进程正常退出、表面吞吐更高，其接触仿真结果也不能用于学习结论。必须先扩大相应物理缓冲区并重新验证完整接触/复位流程；本次没有修改物理配置。

### 建议的加速命令

用于新建实验；启动前检查 GPU 是否有其他作业。下列入口从头训练，不自动接续旧 checkpoint。当前建议先跑 100 轮进行阶段性评估，已有实测结果见第 10 节。

```bash
cd /data/nju/Lincao/InsertAnything_angle

./scripts/python.sh scripts/train_angle_rl.py \
  --task InsertAnythingAngle-LHole-III-Direct-v0 \
  --num_envs 2048 \
  --headless \
  --seed 42 \
  --max_iterations 100 \
  --horizon 128 \
  --seq_length 128 \
  --minibatch_size 8192 \
  --initial_tilt_deg 10 \
  --angle_noise_std_deg 0 \
  --align_weight 1 \
  --align_sigma_deg 5 \
  --learning_rate 0.0001 \
  --run_name "LHole_III_angle0_axispenalty_fast_$(date +%Y%m%d_%H%M%S)"
```

初始夹爪仍竖直向下，插销在手偏角 ±10°，执行时开放 roll/pitch；无触觉、无新增角度噪声。每一步在原奖励上加 `exp(-(alpha/5°)^2)-1`。

如需使用更大的训练批量，可把 `--minibatch_size 8192` 改为 `--minibatch_size 65536`。短测显存增至约 12.3 GiB，吞吐只增加约 5%。在 4 mini epochs 下，actor 每轮优化步数从 128 降到 16，PPO 更新统计也会变化，因此验证奖励时优先用 8,192。不要仅为占满显存而扩大网络或批量；当前短测主要耗时仍是环境采样。

原奖励对照使用相同命令，只改 `--align_weight 0`，并把 run_name 前缀改为 `LHole_III_angle0_control_fast_`。两组应保持并行环境、minibatch、训练轮数等一致，再按第 6 节评估。`run_reward_study.py` 当前不提供 minibatch 覆盖参数；不能直接把这里的 `--minibatch_size` 传给它。

### 训练预算与耗时解释

- 原命令：128 × 128 × 1,000 = **16,384,000** 个环境转移。
- 当前加速命令：2,048 × 128 × 100 = **26,214,400** 个环境转移，是原预算的 **1.6 倍**。500 轮则为 131,072,000，是原预算的 8 倍。
- 两轮短测总吞吐约提升 **13.3 倍**，但完整训练未达到短测速度：用户这次 100 轮实测 **8,030.7 秒，约 2 小时 14 分钟**，包含初始化的平均吞吐约 3,264 环境转移/秒。此前 100 轮约 1 小时、500 轮约 5 小时均为短测外推，不能作为完整训练承诺。
- 保留 1,000 轮会采集原来的 16 倍数据，按同一短测外推约 10 小时/组，不能理解成总训练时间缩短 13 倍。
- 等采样量约需 63 轮，但并行量和优化批量已经变化，等采样量不保证同等收敛，不能把短训练直接视为充分训练。

汇总及原始记录：

- `/data/nju/Lincao/InsertAnything_angle/outputs/throughput_report_20261013.json`
- `outputs/throughput_scaling_20261013_132054/`：512/1,024 环境。
- `outputs/throughput_large_batch_20261013_132444/`：2,048 环境通过；4,096 环境标记为 invalid_physics。
- `outputs/throughput_vram_20261013_132959/`：2,048 环境、65,536 minibatch。

每项基准只训练两轮，未满 299 步完整回合；`rew_-inf` 权重名是尚无已完成回合回报的占位值，不能用于奖励效果比较。吞吐表不替代完整学习曲线与冻结策略评估。


## 10. 2026-10-14：新增轴线惩罚、2,048 环境 × 100 轮的冻结评估

已完成训练目录：`/data/nju/Lincao/InsertAnything_angle/logs/rl_games/InsertAnythingAngle/LHole_III_angle0_axispenalty_fast_20261013_134720`。

训练为原观测＋真实在手角度、零附加角度噪声、±10° 初始在手偏角、初始夹爪竖直、六自由度动作；奖励为原奖励＋`exp(-(alpha/5°)^2)-1`。seed=42，2,048 环境，horizon=128，minibatch=8,192，100 轮，共 26,214,400 个环境转移。实际从 2026-10-13 13:47 运行到 16:01，约 2 小时 14 分钟。

分别冻结最终第 100 轮和最佳训练回报第 99 轮。每个权重使用新种子 1001/2002，每个种子 128 环境，分别运行确定性均值动作与随机动作；每种模式 256 个完整回合，两权重共 **1,024 回合**，每回合 299 控制步（约 19.93 秒）。评估恢复各自完整训练配置，源文件哈希与训练快照一致。

| 权重轮数 / 选择 | 动作模式 | 严格成功 | 回合结束仍严格成功 | 平均轴线夹角：初始 → 结束 | 结束夹角 P95 |
| --- | --- | --- | --- | --- | --- |
| 100 / 最终 | 确定性均值动作 | 103/256 (40.23%) | 103/256 | 4.789° → 2.423° | 8.487° |
| 100 / 最终 | 随机采样动作 | 212/256 (82.81%) | 212/256 | 4.869° → 0.745° | 3.109° |
| 99 / 最佳训练回报 | 确定性均值动作 | 106/256 (41.41%) | 106/256 | 4.789° → 2.266° | 8.174° |
| 99 / 最佳训练回报 | 随机采样动作 | 204/256 (79.69%) | 204/256 | 4.877° → 0.876° | 3.284° |

本次原成功计数与严格成功计数相同。严格成功要求原插入条件满足，并且真实物体/孔轴夹角 <3°、深度误差 ≥−2 mm；没有放宽成功条件或截短失败回合。

### 结果解释

- **策略已经学到纠偏和插入，但随机模式明显优于确定性模式。** 以最终权重为主结果，随机模式平均实际在手角度绝对值由 4.87° 变为 4.12°，结束时 TCP 平均倾斜 4.21°，物体/孔轴误差只有 0.74°。物体仍可以偏在夹爪中，由夹爪整体倾转完成补偿。
- 确定性失败 153 回合，其中 149 回合结束时高于目标插入深度 5 mm 以上；失败组末端深度误差中位数 25.03 mm。另有 75 个失败回合的末端轴线夹角已 <3°，说明对准角度并不等于完成插入。
- 确定性失败中 75 回合最后 5 秒 TCP 的三维包围盒对角线小于 1 mm，存在近乎停滞。随机模式下该计数为 0，成功率也更高。这支持“策略仍依赖随机动作完成搜索/脱离停滞”的解释，但尚未通过改变探索强度的控制实验确认原因。
- 训练日志末段成功率约 76.9% 是训练过程中的随机策略统计，不应拿它直接替代确定性冻结评估结果。模式之间按独立完整回合统计，不把相同 seed 字面值当作完全相同物理初始状态的严格配对。
- **尚不能证明新增奖励优于原奖励。** 同观测、同预算的 `align_weight=0` 对照没有完成。历史原版 26 维观测策略的种子 1001 结果约 84%～85%，其观测、训练预算和优化设置不同，只能作为参考。不能通过两者总回报数值判断奖励优劣。

### 数据核验与文件

所有 306,176 条逐步记录通过有限值、零角度噪声、上一时刻角度观测对应关系、奖励逐项相加、惩罚公式和真实轴线几何检查。有效性检查详见 `data_audit.json`。两项评估日志无 PhysX 接触缓冲溢出、接触点丢弃或 CUDA OOM。

结果目录：`/data/nju/Lincao/InsertAnything_angle/outputs/eval_axispenalty_fast100_20261014/`。

- `analysis.json`：成功率、95% Wilson 区间、角度/位置误差、初始角度分箱、失败诊断与实验限制。
- `final/report.json`、`best/report.json`：每个权重的完整评估与 checkpoint SHA256。
- `final/`、`best/` 下各 `MODE_seed_SEED/diagnostics.csv`：每步角度、成功判定、力和奖励。
- 同目录 `states.npz`：TCP/物体/孔位姿、关节位置速度、原始/裁剪/滤波动作、实际策略观测、奖励及诊断量。
- `evaluation_diagnostics.png`：成功率、角度变化、回合分布与选定成功示例；选定示例不代替整体成功率。
- `training_learning_curves.png`、`training_scalars.json`：本次完整训练曲线。

复现最终权重评估（创建新输出目录）：

```bash
cd /data/nju/Lincao/InsertAnything_angle
./scripts/python.sh scripts/eval_angle_policy.py \
  --checkpoint logs/rl_games/InsertAnythingAngle/LHole_III_angle0_axispenalty_fast_20261013_134720/nn/last_InsertAnythingAngle_ep_100_rew_914.4253.pth \
  --num_envs 128 --headless \
  --seeds 1001 2002 \
  --modes deterministic stochastic \
  --output "outputs/eval_axispenalty_repeat_$(date +%Y%m%d_%H%M%S)"
```

评估最佳权重时将 checkpoint 文件名换为 `InsertAnythingAngle.pth`，其保存轮数为 99；使用新输出目录。暂未重新训练或更改奖励/网络。


## 11. 2026-10-14：与用户已训练的原奖励策略比较

已补齐原版最终第 1,000 轮与最佳第 846 轮的同设置完整评估；连同上节两权重共 2,048 回合。最终权重随机成功率原版 81.64%、本版 82.81%，确定性原版 78.13%、本版 40.23%。两组观测、minibatch、学习率和实际更新数不同，不能视为只改变奖励的消融。完整奖励分解、训练配置证据与下一步改进建议见[现有策略奖励对比](README_insertanything_reward_comparison_20261014.md)。原版已经有深度奖励，优先候选是增加有效插入进度信号；本次没有修改奖励或重新训练。


## 12. 2026-10-15：新增可选深度倾角门控

已按用户的新公式实现 `r_total = r_original + r_axis + (m_theta-1)*r_z_old`。第一版门控为 3° 内取 1，超出后以 5° 宽度平滑衰减至 0.1；原力惩罚和阶段项保留。新训练参数、日志字段、13/13 数学检查、110/110 物理检查、3 轮短训练及完整回合重载记录见[深度倾角门控运行手册](README_insertanything_depth_axis_gate.md)。后续 100 轮正式训练的完整评估见第 13 节。


## 13. 2026-10-15：深度倾角门控版本正式评估

新 run `LHole_III_angle0_depthgate_20261015_005747` 已完成 100 轮。最终权重确定性成功 111/256（43.36%），随机 204/256（79.69%）；旧 Angle 分别为 40.23% 和 82.81%。新最佳回报第 99 轮分别为 40.23% 和 76.95%。确定性角度误差有所减小，但有 97 个失败回合已 <3°，近乎停滞的失败仍为 75 个。完整配置对照、1,024 回合新数据及复现命令见[深度倾角门控手册第 7 节](README_insertanything_depth_axis_gate.md)。
