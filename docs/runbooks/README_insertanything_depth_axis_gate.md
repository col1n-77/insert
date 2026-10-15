# InsertAnything Angle：深度奖励的倾角门控

更新：2026-10-15。用户提出在已有轴线对准项之外，给深度奖励增加 `m_theta(alpha)`。已在独立 `/data/nju/Lincao/InsertAnything_angle` 中实现可选版本 `axis_depth_gated_v1`；旧默认仍为 `axis_additive_v2`。无触觉，Actor/Critic 为 27/74 维，实际在手角度不加噪声，初始夹爪竖直、物体在手偏角 ±10°，执行开放六自由度。

[返回总索引](README.md) · [Angle 训练与评估](README_insertanything_angle_rl.md) · [上一轮现有模型比较](README_insertanything_reward_comparison_20261014.md)

## 1. 实际奖励定义

```text
r_z_old = w_z * m_xy(d_xy) * m_R(d_R) * r_z(d_z)
r_axis  = w_theta * (exp(-(alpha / sigma_axis)^2) - 1)
r_total = r_original + r_axis + (m_theta(alpha) - 1) * r_z_old
```

最后一项把原深度项替换为 `m_theta * r_z_old`，不是重复加入一份深度奖励。`r_original` 包含原 XY、yaw、深度、动作、动作变化、阶段奖励及原力惩罚。用户新公式没有列出力惩罚，本实现继续保留，避免同时改变另一个变量。

- `alpha`：仿真中真实物体轴线与孔轴线夹角；不使用在手角度绝对值或 TCP 倾角。
- `d_R`：当前原代码的对称性处理后的 yaw 误差。倾角由 `alpha` 单独描述。
- `m_xy`：原连续 XY 门控；`m_R`：原 yaw <5° 门控（LHole-III）。
- `r_axis`：沿用此前用户确认的零点形式 `exp(...)-1`，默认权重 1、sigma=5°。
- 原接近、半插入、成功阶段条件及终止规则继续使用原实现。本版本只门控连续深度项，没有把轴线门控同时乘到阶段奖励上。

直接相加本身是有效的奖励组合方式。新形式增加了轴线与深度项的耦合，让倾角较大时的下插收益降低；是否改善确定性策略需要训练后评估，不能仅由公式判定。

## 2. 第一版软门控

```text
excess  = max(alpha - 3 degrees, 0)
m_theta = 0.1 + 0.9 * exp(-(excess / (5 degrees))^2)
```

| 物体/孔轴线夹角 | 深度奖励保留比例 |
| --- | --- |
| 0～3° | 100% |
| 5° | 86.69% |
| 8° | 43.11% |
| 10° | 22.68% |
| 15° | 10.28% |

3° 内保留完整深度奖励，与上一轮严格成功的角度阈值相对应；超过后平滑下降，保留 10% 下限。这些参数是待验证初值。选择软门控是为了保留较大倾角时的下插学习信号，并允许轻微倾角下的接触调整。

若直接用 `exp(-(alpha/5°)^2)`，3° 时深度奖励就只剩 69.77%，10° 时剩 1.83%。此前失败已有不少小角度停滞，因此第一版没有采用这种更强的抑制。阶段奖励尚未附加角度条件，不能把本版本解释为“所有插入相关奖励都必须先严格对准”。如果后续轨迹证明确有倾斜取得阶段奖金的情况，再独立比较给阶段指标添加轴线条件的版本，避免同时改变多项奖励。

## 3. 固定旧轨迹复算：能说明什么

对上一轮 Angle 最终权重的两种动作模式、共 512 回合固定轨迹重新计算新奖励，没有重新执行策略。确定性失败回合最后 5 秒的平均每步总奖励为：

| 评分方式 | 平均每步总奖励 |
| --- | --- |
| 原轴线附加奖励版本 | +1.43487 |
| 本次 3° 容差、10% 下限软门控 | +1.39121 |
| 无容差、无下限的纯 Gaussian 门控 | +1.28131 |

在 `alpha <= 3°` 的每个旧状态，新门控严格等于 1，奖励完全不变。上一轮 153 个确定性失败中有 75 个末端已 <3°，因此本次门控不能直接消除它们在小角度区域的停滞收益。它也不额外奖励向目标深度推进；后续若仍出现这类失败，可另行对照插入进度塑形。

这只是**同一条轨迹改用另一公式评分**，不是新策略测试，不预测新成功率。轨迹和已有训练模型均保留。

## 4. 配置、日志与兼容性

新增配置位于 `/data/nju/Lincao/InsertAnything_angle/source/insertanything_angle/env_cfg.py`：

```yaml
depth_axis_gate:
  enabled: false
  tolerance_deg: 3.0
  sigma_deg: 5.0
  min_gate: 0.1
```

训练时显式传 `--depth_axis_gate` 才启用。只设置门控阈值而不传开关不会启用。旧 checkpoint 的保存配置没有该字段时，评估脚本明确补为关闭，保持旧奖励形式。`--align_weight 0` 仅关闭轴线附加项；若深度门控开关仍开着，深度项仍受门控，不能把它误作纯原奖励对照。

训练配置和 `run.json` 保存具体门控参数及 reward_variant。评估从训练配置自动恢复，不提供静默修改奖励的评估参数。新增逐步字段：

- `depth_axis_gate`：该步门控系数。
- `depth_reward_before_axis_gate`：原有 XY/yaw 门控后的加权深度项。
- `depth_reward_after_axis_gate`：再乘倾角门控后的深度项。
- `depth_gate_reward_adjustment`：本次替换对总奖励的增量，非正。
- `original_reward`：完整原 InsertAnything 奖励，既不含轴线新增项，也不含深度门控增量。
- `depth_gate_adjustment_return`：评估报告中的回合累计门控增量。

TensorBoard 的原 `logs_rew_z_insert` 仍为原始、未加轴线门控的深度分量；实际加权后新深度项使用 `logs_rew_depth_after_axis_gate`，另有 `logs_rew_depth_gate_adjustment` 和 `logs_angle_depth_axis_gate`，避免混用。

## 5. 验证记录

结果目录：`/data/nju/Lincao/InsertAnything_angle/outputs/depth_gate_revision_20261015`。

- 数学检查 **13/13**：有界单调门控、容差内恒等、边界连续、参数校验、夹爪倾转补偿后物体对准时门控仍为 1，以及既有几何/噪声测试。
- **128 环境 × 8 步物理检查，110/110 通过**：六任务配置继承、门控开启和关闭时的奖励组合、关闭时与旧奖励精确一致、权重 0 的语义、原奖励逐项保留、零角度噪声及六自由度初始化。六任务均核对配置，实际物理 rollout 为 LHole-III。
- 原始实现备份和哈希：`source_before/`、`change_manifest.json`。
- `physical_contract.json` / `.log`：短时物理检查结果。
- `fixed_trajectory_rescoring.json` / `.log`、`replay_gate_rewards.py`：固定旧轨迹复算记录。

16 环境 × 3 轮 PPO 冒烟训练已完成（6,144 个环境转移，约 83.9 秒），run 为 `logs/rl_games/InsertAnythingAngle/depth_gate_contract_smoke_20261015`。重载该短训练权重后，两种动作模式各 4 个完整 299 步回合均通过配置恢复、有限值、门控和奖励相加检查；逐步总奖励最大复算误差约 5.97e-8。两个模式均成功 0/4，这是仅训练 3 轮的链路检查，不能用于判断门控好坏。新日志标签均已写入 TensorBoard，汇总验收见 `validation.json`。

短检查只说明实现和运行链路可用，不证明新奖励提升成功率。这里记录的是实现检查；后续 100 轮正式训练的完整评估见第 7 节。

## 6. 训练与评估命令

前置条件：同级原版 `.venv-isaac51`、原任务代码和资产仍存在，GPU 可用。先运行有限物理检查，输出文件须尚不存在：

```bash
cd /data/nju/Lincao/InsertAnything_angle
./scripts/python.sh scripts/check_angle_env.py \
  --headless --num_envs 128 --steps 8 --depth_axis_gate \
  --output "outputs/depth_gate_check_$(date +%Y%m%d_%H%M%S).json"
```

复现已评估配置的训练命令（会从头创建新实验）：

```bash
cd /data/nju/Lincao/InsertAnything_angle
./scripts/python.sh scripts/train_angle_rl.py \
  --task InsertAnythingAngle-LHole-III-Direct-v0 \
  --num_envs 2048 --headless --seed 42 \
  --max_iterations 100 --horizon 128 --seq_length 128 \
  --minibatch_size 8192 \
  --initial_tilt_deg 10 --angle_noise_std_deg 0 \
  --align_weight 1 --align_sigma_deg 5 \
  --depth_axis_gate \
  --depth_gate_tolerance_deg 3 \
  --depth_gate_sigma_deg 5 \
  --depth_gate_floor 0.1 \
  --learning_rate 0.0001 \
  --run_name "LHole_III_angle0_depthgate_$(date +%Y%m%d_%H%M%S)"
```

100 轮是与已有 Angle 最终权重匹配的首轮比较预算，不是收敛保证。该入口从头训练，不会加载旧 checkpoint。与已有 2,048 环境、100 轮、seed=42、minibatch=8,192 的 Angle 模型比较时，可针对本次深度门控进行对照；原版 26 维模型仍只能作为不同系统的参考。

同配置对照只去掉 `--depth_axis_gate`，其余配置一致并使用不同 run_name；若还要纯原奖励对照，则同时设 `--align_weight 0`。`run_reward_study.py` 仍是此前附加奖励的双组队列，不接收本次门控参数。

训练完成后从终端 `ANGLE_RL_OUTPUT` 获取真实目录，选择该 run 的实际 checkpoint，再按以下模板评估；`CHECKPOINT_PATH` 必须替换为存在的文件：

```bash
cd /data/nju/Lincao/InsertAnything_angle
./scripts/python.sh scripts/eval_angle_policy.py \
  --checkpoint CHECKPOINT_PATH \
  --num_envs 128 --headless --seeds 1001 2002 \
  --modes deterministic stochastic \
  --output "outputs/eval_depthgate_$(date +%Y%m%d_%H%M%S)"
```

同时报告确定性和随机模式的严格成功率、末端轴线误差、深度误差、首次成功时间和停滞比例。评估门控会从训练配置自动恢复；不通过修改旧模型评估时的奖励来模拟重新训练。


## 7. 2026-10-15：100 轮正式训练的完整评估

用户完成的 run：`/data/nju/Lincao/InsertAnything_angle/logs/rl_games/InsertAnythingAngle/LHole_III_angle0_depthgate_20261015_005747`。2,048 环境、100 轮、26,214,400 个环境转移、seed=42，实际训练约 4,436.6 秒（1 小时 14 分钟）。本次评估最终第 100 轮和最佳训练回报第 99 轮；每个权重分别运行确定性/随机动作，各使用 seeds 1001、2002 × 128 环境。新增 **1,024 个完整回合、306,176 条逐步记录**，每回合 299 控制步，约 19.933 秒。

主结论：本次门控使确定性末端平均轴线误差减小，但成功率没有表现出稳定提升。最终权重确定性相对旧 Angle 增加 8 个成功回合，随机模式减少 8 个；两个模式的 95% Wilson 区间均与旧 Angle 重叠。最佳训练回报权重的补充评估也未显示成功率优势。以上限于一个训练种子，不能推断所有训练种子的结果。

### 最终权重主比较

| 策略 | 动作模式 | 严格成功 | 95% Wilson 区间 | 末端平均轴线误差 | 末端误差 P95 |
| --- | --- | --- | --- | --- | --- |
| 原奖励，最终 1000 轮 | 确定性 | 200/256（78.13%） | 72.67%～82.75% | 0.843° | 2.800° |
| 原奖励，最终 1000 轮 | 随机 | 209/256（81.64%） | 76.44%～85.90% | 1.212° | 3.123° |
| 仅轴线奖励，最终 100 轮 | 确定性 | 103/256（40.23%） | 34.41%～46.34% | 2.423° | 8.487° |
| 仅轴线奖励，最终 100 轮 | 随机 | 212/256（82.81%） | 77.71%～86.94% | 0.745° | 3.109° |
| 轴线奖励＋深度门控，最终 100 轮 | 确定性 | 111/256（43.36%） | 37.43%～49.48% | 1.630° | 5.322° |
| 轴线奖励＋深度门控，最终 100 轮 | 随机 | 204/256（79.69%） | 74.34%～84.16% | 0.746° | 2.364° |

原成功条件与严格条件计数相同，所有本次成功回合在结束时仍满足严格成功。严格条件为原任务成功条件，加真实物体/孔轴夹角 <3°、深度误差 ≥−2 mm。角度统计包含全部成功和失败回合。

### 最佳训练回报权重补充比较

| 策略 | 动作模式 | 严格成功 | 95% Wilson 区间 | 末端平均轴线误差 | 末端误差 P95 |
| --- | --- | --- | --- | --- | --- |
| 仅轴线奖励，最佳回报 99 轮 | 确定性 | 106/256（41.41%） | 35.54%～47.52% | 2.266° | 8.174° |
| 仅轴线奖励，最佳回报 99 轮 | 随机 | 204/256（79.69%） | 74.34%～84.16% | 0.876° | 3.284° |
| 深度门控，最佳回报 99 轮 | 确定性 | 103/256（40.23%） | 34.41%～46.34% | 1.836° | 5.786° |
| 深度门控，最佳回报 99 轮 | 随机 | 197/256（76.95%） | 71.42%～81.69% | 0.834° | 2.950° |

当前评估中，新版最终第 100 轮的两种动作模式均优于其最佳训练回报第 99 轮；后续使用本次新模型时可优先选择最终权重，不把训练回报最高自动等同于插入成功率最高。

### 对照条件核对

新旧 Angle 均为 27/74 维 Actor/Critic、无角度观测噪声、初始在手偏角 ±10°、初始夹爪竖直、六自由度动作；均使用 2,048 环境、horizon/sequence=128、minibatch=8,192、固定学习率 1e-4、100 轮和 seed=42。保存的 env.yaml 仅新增 depth_axis_gate，agent.yaml 仅实验名称不同；实际 actor Adam 更新数分别为 12,796 和 12,795，训练预算基本一致。

确定性两种子和随机 seed=2002 的初始物体位姿完全匹配；随机 seed=1001 最大位置差约 0.0051 mm、平均轴线误差差约 0.0000134°。因此这次比此前“26 维原版对 27 维 Angle”更适合检查门控影响，但仍只比较一个训练种子。原版 26 维参考策略的训练预算、学习率等不同。

原版与旧 Angle 的 2,048 回合结果直接复用此前冻结报告，没有再次运行或更改。加上本次结果，对比 JSON 共包含六个权重、3,072 回合。

### 停滞与奖励诊断

| 最终权重的确定性失败诊断 | 旧 Angle，无门控 | 新 Angle，有门控 |
| --- | --- | --- |
| 失败回合 | 153 | 145 |
| 失败中末端轴线误差 <3° | 75 | 97 |
| 失败中距目标深度 >5 mm | 149 | 139 |
| 失败组末端深度误差中位数 | 25.03 mm | 24.69 mm |
| 失败中最后 5 秒 TCP 运动范围 <1 mm | 75 | 75 |

运动范围定义为最后 5 秒 TCP 三维包围盒对角线。两组停滞数量相同，不代表完全相同的环境编号；已对准和停滞的失败子集也可以重合。

新版确定性失败片段最后 5 秒每步仍平均获得 1.909 的总奖励。连续深度项在门控前为 +0.504，门控调整仅 -0.018，轴线项为 -0.252，接近/半插入阶段合计 +0.587。原 XY/yaw 正奖励继续存在。不同模型的失败集合不同，不能把这组均值差异当成严格的同状态奖励对照。

新版全轨迹的平均门控系数为 0.970；进入 3° 容差内时门控严格等于 1。已有 97 个失败回合末端满足该角度要求；这里的“<3°”只是奖励门控容差，不代表满足 L 形孔的精配合要求。第 8 节进一步记录了持续下压、接触卡滞和受控诊断结果。现有证据不支持仅凭继续加强倾角门控就能解决停滞；若后续改动，应单独测试有效插入进度信号，并继续保持训练预算一致。本次评估没有改奖励或启动新的正式训练。

![最终权重成功率、对准、插入进度及失败分布](/data/nju/Lincao/InsertAnything_angle/outputs/eval_depthgate_fast100_20261015/policy_comparison.png)

![失败片段奖励贡献与门控系数](/data/nju/Lincao/InsertAnything_angle/outputs/eval_depthgate_fast100_20261015/reward_diagnostics.png)

第二张图中 depth 表示原加权深度项，gate 表示新增调整，二者相加才是实际深度奖励；stages 为接近、半插入及成功阶段项之和。

### 验收与结果目录

结果目录：`/data/nju/Lincao/InsertAnything_angle/outputs/eval_depthgate_fast100_20261015`。

- `final/report.json`、`best/report.json`：完整评估、每回合指标和 checkpoint SHA256，模型快照保存在各自目录。
- 各模式/种子下的 `diagnostics.csv`、`states.npz`、`reset.json`：逐步奖励、门控、物体/TCP/孔位姿、关节状态、力及动作。
- `comparison.json`：六个权重的统一结果、初始姿态核对、失败诊断、奖励分解和配置差异。
- `data_audit.json`、`runtime_validation.json`：8 个新批次全部通过，306,176 条记录有限；根据姿态复算轴线、成功条件、角度观测对应关系、原奖励和门控。总奖励最大复算误差 1.14e-06。
- `analyze_evaluation.py`、`evaluation_script.py`、`training_source_manifest.json`：复算脚本及来源快照；训练的 23 项源文件哈希匹配当前实现。
- `training_scalars.json`：新旧 Angle 的训练标量；训练阶段记录不能代替完整回合评估。

评估日志没有 PhysX 接触缓冲溢出、接触点丢弃、CUDA OOM 或异常栈。两张图已检查显示。主分析基于物理成功率，不比较不同奖励定义下总回报的绝对高低。

复现最终权重评估（输出目录须尚不存在）：

```bash
cd /data/nju/Lincao/InsertAnything_angle
./scripts/python.sh scripts/eval_angle_policy.py \
  --checkpoint logs/rl_games/InsertAnythingAngle/LHole_III_angle0_depthgate_20261015_005747/nn/last_InsertAnythingAngle_ep_100_rew_906.8544.pth \
  --num_envs 128 --headless --seeds 1001 2002 \
  --modes deterministic stochastic \
  --output "outputs/eval_depthgate_repeat_$(date +%Y%m%d_%H%M%S)"
```

评估第 99 轮最佳训练回报权重时，把 checkpoint 文件名换为 `InsertAnythingAngle.pth`，并使用新输出目录；门控参数会从训练配置恢复。


## 8. TCP 停滞的控制与接触诊断（2026-10-15）

当前证据表明：确定性策略仍然持续要求下压，TCP 在接触约束下停住；残留的 XY、yaw、轴线误差与有限推力共同影响能否继续插入。不能把停滞直接解释为“策略因为奖励为正而不再输出向下动作”。奖励可能影响精细纠偏的学习，但本轮诊断不构成奖励最优性证明。

### 已验证的事实

1. 对此前 256 个确定性回合，75 个失败回合最后 5 秒 TCP 三维运动范围 <1 mm。这 75 个回合的所有末段 Z 动作均向下，均值 -0.9066；53 个回合的平均 Z 动作 <-0.95。末段 TCP 净 Z 位移均值仅 -0.0710 mm。
2. 使用相同最终权重、128 环境、seed=1001，增加只读控制记录。全部 299 步 TCP、物体旋转、动作及奖励与此前评估逐值相同。该种子有 40 个上述停滞失败，末段平均向下控制力命令 3.286 N，中位数 3.223 N；平均目标在 TCP 下方 18.176 mm。
3. 这 40 个回合末段 Z 位置目标被边界裁剪的比例为 0，Z 力命令被死区清零的比例为 0，最大机械臂关节力矩命令约 3.36 N·m，未触发该机器人的力矩上限。没有证据表明此处 Z 动作被执行代码锁住。
4. 控制是力矩式 PD：`target_z = current_tcp_z + 0.02 * filtered_action_z`，每个物理子步重新生成。动作 -1 对应当前 TCP 下方 20 mm 的控制目标，并非保证每步实际下降 20 mm。Kp_z 随机在 100～300 N/m，静止时比例项约 2～6 N；目标不累计推进，接触后也不会自动持续增力。
5. `depth_axis_gate` 只乘到奖励，不作用于执行动作。75 个停滞失败的末段门控平均仍为 0.9725。仅提高这个门控的强度不能直接增加下压力。

这里的控制力为控制器要求的 task-space wrench，不是实测净接触力。原 `contact_force_local_n` 是经过夹持力去零和 EMA 的信号，不能把它与控制力直接等同，也不能因该值较小就排除接触。

### 几何上为什么还会卡

直接读取原 USD 三角网格截面，L 孔外廓宽 15.0 mm、插销外廓宽 14.9 mm；完美对齐时最小单侧间隙约 0.050 mm。孔壁高度为 25 mm，轴线倾角 3° 在该长度上对应约 1.310 mm 的横向变化。因此 3° 门控容差明显宽于几何配合尺度；L 形还必须同时修正 yaw 和 XY。

这些是原网格的名义几何量，不是可直接拿来替代 PhysX 的硬成功阈值；实际接触使用 SDF，存在离散与求解容差。

| seed=1001 的失败环境 | 最后轴线误差 | 最后 yaw 误差 | 最后 XY 误差 | 剩余深度 | 深度轴线门控 |
| --- | --- | --- | --- | --- | --- |
| 3：孔口附近受阻 | 1.147° | 0.175° | 4.764 mm | 25.193 mm | 1 |
| 8：部分插入后停住 | 1.139° | 0.451° | 0.189 mm | 7.965 mm | 1 |
| 28：轴线很小仍卡住 | 0.167° | 0.584° | 0.029 mm | 6.412 mm | 1 |

环境 28 说明仅把轴线角压小还不够；剩余 yaw、微小横向误差、摩擦和接触状态仍会影响插入。轴线 <3° 的失败不应称为“已经精确对准”。

![TCP、下压力、深度和角度的受控轨迹](/data/nju/Lincao/InsertAnything_angle/outputs/tcp_stall_diagnosis_20261015/tcp_stall_case8.png)

[两种典型失败的完整曲线](/data/nju/Lincao/InsertAnything_angle/outputs/tcp_stall_diagnosis_20261015/tcp_stall_traces.png) · [实际网格截面对比](/data/nju/Lincao/InsertAnything_angle/outputs/tcp_stall_diagnosis_20261015/peg_hole_sections.png)

### 短时受控干预

三组均使用同一权重、seed=1001、128 环境，前 225 步（15 秒）TCP、物体姿态、动作和奖励逐值相同。第 226 步开始干预，到原 20 秒回合结束，严格成功统计如下：

| 后 5 秒处理 | 成功数 | 相对原确定性新成功 | 相对原确定性失去的成功 |
| --- | --- | --- | --- |
| 继续确定性，控制不变 | 57/128 | — | — |
| Kp_z ×2，阻尼不变 | 66/128 | 9 | 0 |
| 原控制器，切换为策略采样动作 | 67/128 | 12 | 2 |

在原 40 个停滞失败上，前者使 8 个成功，后者使 9 个成功。切换采样后的平均向下控制力反而较小（2.729 N，原确定性 3.286 N），但部分环境可通过新的动作序列脱困。加大刚度也能救回一部分回合，说明接触下的推动能力确有影响；仍有多数失败没有被这两种简单处理解决。

这是单一评估种子上的因果诊断，不是新训练策略的正式性能比较。加刚度会进一步改变闭环姿态和后续动作；采样改变六个动作通道及后续观测噪声序列，因此不能把它简化为仅某一轴抖动的效果。诊断使用独立运行钩子，没有修改任何生产控制器、奖励或训练配置。

### 对后续奖励/训练的含义

- 原 XY 奖励和新增轴线项在精配合尺度上的分辨能力有限：例如 XY 误差 0.2 mm 时，原 `w_xy*r_xy` 约 0.624984，最大值为 0.625；`exp(-100*d_xy)` 深度门控仍约 0.9802。3° 内新增轴线深度门控完全放行，原 yaw 门控也是较宽的 5°。
- 后续应重点验证孔口附近的 XY/yaw/轴线联合精对准，以及真实有效插入进度信号，使策略学到卡住后的纠偏、适度卸载和重新下插。只根据“向下动作已经发出”加奖励会奖励持续顶住。
- 零噪声只针对新增在手角度通道。保存配置中仍有原孔位静态噪声标准差 1 mm、孔位/TCP 位置动态噪声各 0.5 mm，以及原姿态、速度、力噪声和动力学随机化。奖励使用真值，不受新增角度噪声影响；策略的观测仍不精确。若要隔离奖励效果，应把原观测噪声作为单独消融变量，不在现有对照中混改。
- 本轮不足以证明“增大角度权重”或“加大 Z 刚度”就是最佳方案；下一轮应按相同训练预算分别验证奖励与控制改动，并保留确定性评估。

### 结果与复现

所有新诊断产物位于 `/data/nju/Lincao/InsertAnything_angle/outputs/tcp_stall_diagnosis_20261015`：

- `diagnosis.json`：两种子停滞动作统计、逐环境指标、控制证据、几何量、受控干预及解释限制。
- `trace/`、`double_z_stiffness/`、`late_stochastic/`：三个冻结权重快照、配置、完整回合报告；每组的 `deterministic_seed_1001/states.npz` 含控制子步位姿误差、死区前后力命令、关节力矩、位置裁剪量等；`controller_reset.json` 保存实际 Kp/Kd、死区、摩擦、动作尺度和观测偏置。
- `diagnose_tcp_stall.py`：独立诊断脚本，复制原评估入口后添加记录和明确的诊断参数；只使用自身进程内钩子。
- `analyze_tcp_stall.py`：离线复算与绘图；`asset_geometry.json` 和 `export_asset_geometry.py` 为原资产截面数据与导出源代码。
- `validation.json`：三组完成状态、有限值、受控前缀一致性、权重 SHA256、23 项生产源文件未改动的核验。

重跑一次记录（输出必须是新目录，需要可访问 GPU 的 Isaac Sim 环境）：

```bash
cd /data/nju/Lincao/InsertAnything_angle
./scripts/python.sh outputs/tcp_stall_diagnosis_20261015/diagnose_tcp_stall.py \
  --training_run logs/rl_games/InsertAnythingAngle/LHole_III_angle0_depthgate_20261015_005747 \
  --checkpoint outputs/eval_depthgate_fast100_20261015/final/evaluated_checkpoint.pth \
  --num_envs 128 --seeds 1001 --modes deterministic --headless \
  --intervention none --intervention_step 226 \
  --output "outputs/tcp_stall_repeat_$(date +%Y%m%d_%H%M%S)"
```

两种诊断干预分别将 `--intervention` 改为 `double_z_stiffness` 或 `late_stochastic`，并使用不同的新输出目录。正式评估继续使用第 7 节的 `scripts/eval_angle_policy.py`。


## 9. 失败视频与同步 Z 动作/力（2026-10-15）

已录制第 8 节的同一个环境：seed=1001、环境编号 8、128 环境并行、确定性策略、第 100 轮最终权重。记录无控制干预，299 帧、15 fps、19.933 秒；录像的 TCP、物体旋转、动作和奖励与前一轮无渲染诊断逐值一致。

- [同步视频：场景＋Z action＋Fz＋TCP/深度曲线](/data/nju/Lincao/InsertAnything_angle/outputs/tcp_stall_video_20261015/failure_env8_synced.mp4)
- [完整静态曲线图](/data/nju/Lincao/InsertAnything_angle/outputs/tcp_stall_video_20261015/z_action_force.png)
- [逐帧 CSV 数据](/data/nju/Lincao/InsertAnything_angle/outputs/tcp_stall_video_20261015/z_action_force.csv)
- [逐帧 NumPy 数据](/data/nju/Lincao/InsertAnything_angle/outputs/tcp_stall_video_20261015/env8_z_signals.npz)
- [无叠加的原始场景视频](/data/nju/Lincao/InsertAnything_angle/outputs/tcp_stall_video_20261015/recording/deterministic_seed_1001/scene.mp4)
- [校验与数值摘要](/data/nju/Lincao/InsertAnything_angle/outputs/tcp_stall_video_20261015/video_summary.json)

重点看约 13～20 秒。最后 5 秒：执行 Z 动作均值 -0.999999，控制器世界 Fz 均值 -5.2635 N，TCP 净下降仅 0.1171 mm，三维运动范围 0.2335 mm；回合最终仍差 7.9652 mm，没有出现成功。

### 力的口径补充核验

本次进一步核对保存配置与已安装 Isaac Lab 源码：ContactSensor 的实际 prim 在 `HeldAsset`，body 为 `square_peg_10mm`。虽然配置名称为 `gripper_peg`、filter 列表指向左右手指，当前环境读取的是 `net_forces_w`，而不是受过滤器约束的 `force_matrix_w`。因此原信号实际上是**插销上所有接触的净法向合力**，不能称为独立的夹爪—插销力或单独的插销—孔力。

- 世界 +Z 向上；负 Z action、负控制器 Fz 表示向下。
- `commanded_fz_world_n_mean`：执行控制器的死区后力命令，8 个物理子步取均值；不是实测力。
- `peg_net_normal_fz_world_n_raw`：插销净法向接触力的世界 Z 分量，读取动作后的最后物理子步；最后 5 秒均值 -0.2066 N。它不包含切向摩擦，不是完整六维力传感器，也不能用它单独做竖直受力平衡。
- `peg_net_normal_fz_world_n_processed`：沿用原环境的去零、物体坐标系 EMA 处理，再旋转表达为世界坐标；另保留原物体坐标 Z 分量。
- CSV 包含原始 Gaussian 均值、裁剪动作、EMA 执行动作、控制器 Fz 均值/最小/最大、上述力信号、TCP Z、剩余深度、轴线/yaw/XY 误差及成功标记。

原始 `states.npz` 中新加字段的历史前缀 `raw_gripper_*` 保留以便溯源，但其实际测量对象为插销；CSV、图及修正后的 report 均使用准确描述。实际执行脚本保存在 `record_tcp_stall_executed.py`，之后仅修正力说明的入口为 `record_tcp_stall.py`。数值没有改写，修正前 report 也已保留。

视频第 i 帧的 PTS 为 i/15 秒，显示的是第 i+1 步结束后的状态，仿真时间为 (i+1)/15 秒；CSV 同时列出两种时间，动态图的红线使用仿真时间。原始片段没有把 reset 截图作为视频第 0 帧。

重新录制（输出目录需不存在）：

```bash
cd /data/nju/Lincao/InsertAnything_angle
./scripts/python.sh outputs/tcp_stall_video_20261015/record_tcp_stall.py \
  --training_run logs/rl_games/InsertAnythingAngle/LHole_III_angle0_depthgate_20261015_005747 \
  --checkpoint outputs/eval_depthgate_fast100_20261015/final/evaluated_checkpoint.pth \
  --num_envs 128 --seeds 1001 --modes deterministic --headless \
  --intervention none --video --video_env 8 \
  --camera_eye 0.82 0.22 0.22 --camera_lookat 0.5707 0.0017 0.065 \
  --output "outputs/tcp_failure_video_repeat_$(date +%Y%m%d_%H%M%S)"
```

合成入口为 `outputs/tcp_stall_video_20261015/package_video.py`。本次运行没有修改生产奖励、控制器或训练模型。


## 10. 剩余 8 mm 与孔壁摩擦的直接测量（2026-10-15）

对第 9 节同一个失败回合增加只读接触探针：seed=1001、环境 8、128 环境、相同最终权重、确定性动作。299 个控制步 × 8 个物理子步，分别读取插销—孔、插销—左指、插销—右指的法向力与切向摩擦力。TCP、物体旋转、执行动作和奖励与原失败录像逐值一致；未改变生产奖励、控制器或训练配置。

### 8 mm 是剩余深度

最终 TCP 世界 Z 为 54.148 mm；插销底部参考点世界 Z 为 21.101 mm；孔底目标世界 Z 为 13.136 mm。图中剩余深度是后两者的差 7.965 mm。孔壁高 25 mm，插销底部参考点已经从孔口进入约 17.035 mm。不存在“到 8 mm 就停止下压”的控制阈值；这是该回合在当前接触姿态和推动能力下停住的位置。

### 末段 5 秒的真实接触分解

以下均为作用在插销上的世界 Z 向力，正值向上，统计全部物理子步：

| 来源 | 平均 Fz（N） |
| --- | ---: |
| 孔壁切向摩擦 | +5.471365 |
| 孔壁法向力的 Z 分量 | +0.007246 |
| 左右手指合计接触力（法向＋摩擦） | -5.262796 |
| 插销重力（质量 0.022 kg） | -0.215820 |
| 合力 | 约 -0.000007 |

该回合停滞时确实主要由孔壁的轴向摩擦抵抗下压。孔壁法向主要沿水平方向，其 Z 分量小并不代表接触弱：各接触点法向力大小之和均值约 23.963 N，而孔壁净法向合力大小的均值仅约 0.865 N，多处接触的方向会相互抵消。这里的摩擦力直接来自 PhysX `get_friction_data(dt=physics_dt)`，不是用摩擦系数乘净法向合力估算。

之前视频中的绿色力曲线只包含插销的净法向接触力，遗漏切向摩擦，不能据此认为轴向阻力很小。控制器 Fz 命令均值 -5.263525 N，与独立测得的手指总向下作用接近；上、下作用近似平衡，所以持续向下 action 仍可对应很小的位移。该结论验证的是这个失败回合，不代表所有失败的成因一致，也不把微小爬移判作严格零速度的静摩擦。

![剩余深度定义与分项受力](/data/nju/Lincao/InsertAnything_angle/outputs/friction_diagnosis_20261015/depth_and_friction.png)

- [完整分项力随时间变化](/data/nju/Lincao/InsertAnything_angle/outputs/friction_diagnosis_20261015/force_decomposition_traces.png)
- [逐物理子步 CSV](/data/nju/Lincao/InsertAnything_angle/outputs/friction_diagnosis_20261015/pairwise_contact_forces.csv)
- [高度、力与校验摘要](/data/nju/Lincao/InsertAnything_angle/outputs/friction_diagnosis_20261015/friction_summary.json)
- 原始数据：`/data/nju/Lincao/InsertAnything_angle/outputs/friction_diagnosis_20261015/pair_force_trace/deterministic_seed_1001/states.npz`。
- 接触过滤器与对象：同目录上一级的 `force_probe_metadata.json`；顺序为孔、左指、右指。每个物理步长为 1/120 秒。
- 接触点法向求和与 PhysX 成对法向矩阵最大差 0.0000058 N；成对法向总和与全体净法向最大差 0.0000010 N；全部数值有限，接触容量未溢出。

复现须有可访问 GPU 的既有 Isaac Sim 运行环境；脚本固定探测环境 8，复现本回合请保持 128 环境及以下种子。输出使用新目录：

```bash
cd /data/nju/Lincao/InsertAnything_angle
./scripts/python.sh outputs/friction_diagnosis_20261015/probe_contact_friction.py \
  --training_run logs/rl_games/InsertAnythingAngle/LHole_III_angle0_depthgate_20261015_005747 \
  --checkpoint outputs/eval_depthgate_fast100_20261015/final/evaluated_checkpoint.pth \
  --num_envs 128 --seeds 1001 --modes deterministic --headless \
  --intervention none \
  --output "outputs/friction_probe_repeat_$(date +%Y%m%d_%H%M%S)"
```
