# 同配置轴线奖励对照

[返回 README](../README.md) · [完整命令](WORKFLOWS.md)

## 实验问题

在无触觉 LHole-III 中，保持输入、复位、动作、物理参数、原奖励及 PPO 预算相同，只比较轴线惩罚权重 0 和 1。两组都观测真实在手偏角，避免同时改变输入维度。新增角度噪声为 0；原版 TCP 观测噪声及动力学随机化保留。

初始 TCP 竖直向下，插销绕 TCP +Y 采样 ±10°，随后物理闭爪；测得角度可与请求角度不同。动作允许三维平移及三维旋转。物体的在手偏角、TCP 世界姿态和物体/孔轴夹角是不同量。

## 奖励与可选门控

```text
alpha = acos(clip(dot(object_axis_world, hole_axis_world), -1, 1))
r_axis = exp(-(alpha / sigma_axis)^2) - 1
r_new = r_original + axis_weight * r_axis
```

计算中角度统一为弧度，命令行 sigma 用度。`sigma_axis=5°`；对齐时新增项为 0，严重失配时趋近 -1。权重 0 组保留角度输入、原力惩罚及原成功定义。

单次训练可另加 `--depth_axis_gate`：

```text
excess = max(alpha - tolerance, 0)
gate = floor + (1 - floor) * exp(-(excess / gate_sigma)^2)
r_new = r_original + axis_weight * r_axis + (gate - 1) * original_dense_depth_reward
```

默认 tolerance=3°、gate_sigma=5°、floor=0.1。这个修正只替换连续深度奖励，不重复增加深度项，也不门控阶段事件奖励。**匹配队列两组均关闭门控**，后续才适合单独研究它的作用。

## 两组共同训练设置

| 参数 | 值 |
| --- | --- |
| task | `InsertAnythingAngle-LHole-III-Direct-v0` |
| Actor / Critic 维度 | 27 / 74 |
| 环境数 / 轮数 | 2048 / 500 |
| horizon / sequence | 128 / 128 |
| minibatch / mini_epochs | 8192 / 4 |
| 训练 seed | 42 |
| 初始在手偏角 / 新增角度噪声 | ±10° / 0° |
| 动作 / 原力惩罚 | 六自由度 / 开启 |
| 深度门控 | 关闭 |
| 初始 LR / Actor 调度 | 1e-4 / adaptive |
| schedule_type / KL 阈值 | standard / 0.008 |
| checkpoint 保存间隔 | 100 轮，另存训练回报最佳权重及最终权重 |

Actor 每个 PPO mini-epoch 按平均 KL 更新：KL >0.016 时 LR 除以 1.5，KL <0.004 时乘以 1.5，限制在 1e-6～1e-2。当前 rl_games 1.6.1 的独立 central critic 对非 linear 调度使用 IdentityScheduler；虽然原版 critic YAML 写 adaptive，实际仍保持 1e-4。本实验保留该原始行为，未新增 critic 调度实现。

同一规则不要求两组逐轮学习率数值完全相同；KL 是各自策略更新的结果。初始 LR、阈值和更新频率相同即可。使用 `--lr_schedule constant` 可显式复现固定 LR；独立 `train_angle_rl.py` 默认 constant，而匹配入口默认 adaptive。

每组计划采样 131,072,000 条转移、Actor 更新 64,000 次。轮数是 PPO epoch，不是完成回合数。500 轮不是效果保证；单个训练 seed 的结果也不足以证明跨训练种子的稳定优势。

## 队列执行和产物

从仓库根目录运行 README 中的匹配命令。`--plan_only` 仅写计划，无仿真、无训练。正式运行使用新的 study_name；所有输出目录必须不存在。

```text
train control
  → evaluate control final
  → evaluate control best_training_reward
  → train aligned
  → evaluate aligned final
  → evaluate aligned best_training_reward
  → summarize
```

```text
InsertAnything_angle/
├── outputs/<study>/
│   ├── study.json                  # 实时阶段、PID、最新轮次、配置检查、错误
│   ├── train_control.log
│   ├── train_aligned.log           # B 组开始时产生
│   ├── configuration_comparison.json
│   ├── eval_control_final/        # 相应评估完成后产生
│   ├── eval_aligned_final/
│   └── comparison.json            # 全部阶段完成后产生
└── logs/rl_games/InsertAnythingAngle/<study>_<arm>/
    ├── run.json
    ├── params/{env,agent}.yaml
    ├── source_snapshot/
    ├── source_manifest.json
    ├── summaries/                  # TensorBoard：info/last_lr、info/kl、info/cval_lr
    └── nn/
```

每组实际保存配置接受 29 项检查，包括维度、从头训练、初始姿态、力惩罚和 Actor/critic 调度。两组完整配置只允许 `axis_alignment.reward_weight` 和 `params.config.full_experiment_name` 不同。匹配器在每个新阶段开始前重新核对源码 SHA-256；训练期间修改被跟踪源码会导致队列停止。

日志出现 PhysX 接触丢弃、CUDA OOM、Python 异常栈，或阶段退出码非零，会中止后续阶段。不要在失败后直接复用同名输出目录；检查原因与已有产物，再选择新的实验名重新执行。本队列没有自动断点续训功能。

启动是前台执行。需要关闭终端后继续时，可在 tmux 中运行；不要重复启动同一个实验。原工作站本轮采用独立后台队列，历史 PID 不应在其他机器执行。

## 查看和停止

从仓库根目录，替换实际 study_name：

```bash
export INSERT_STUDY_DIR="$PWD/InsertAnything_angle/outputs/your_study_name"
cat "$INSERT_STUDY_DIR/study.json"
tail -f "$INSERT_STUDY_DIR/train_control.log"
```

Linux 下停止整个队列，先核对状态及 PID 所属命令：

```bash
python3 - <<'PY'
import json, os, signal
from pathlib import Path
s = json.loads((Path(os.environ['INSERT_STUDY_DIR'])/'study.json').read_text())
if s['status'] != 'running':
    raise SystemExit('The queue is not running; inspect study.json first.')
pid = s['pid']
cmd = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
assert any(b'run_matched_reward_study.py' in x for x in cmd)
assert s['arguments']['study_name'].encode() in cmd
os.kill(pid, signal.SIGTERM)
PY
```

运行器会终止它启动的当前子进程并保留已保存文件，不再启动下一组。直接在前台 Ctrl-C 也可触发这一处理。

## 比较规则

每份冻结权重：128 环境 × 2 个新 seed（1001、2002）× 2 种动作模式，共 512 回合。最终和最佳权重均存在时，两组共评估 2048 回合。评估 seed 不得与训练 seed 重复。

主比较采用两组最终第 500 轮权重；最佳训练回报权重为补充，其被选中的轮数可能不同。报告包括原成功率、严格成功率、终态轴线误差、XY/深度误差及动作饱和比例。严格成功是原成功条件再加轴线误差 <3°、剩余深度不小于 -2 mm；同时保留终态与整个回合曾成功两类指标。

确定性取策略均值，随机模式从分布采样；二者均需要评估。不要将平均训练 reward 直接跨奖励定义排序，也不要把静态触觉角度 MAE 当作插入成功率。

2026-10-17 的工作站训练先以固定 LR 启动，control 第 23 轮停止；随后用本页 adaptive 配置重新从头训练。本发布快照尚没有两组均完成的正式结果。旧权重、日志和审计记录继续保留在工作站，未包含在 Git 仓库。
