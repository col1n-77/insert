# InsertAnything：触觉在手偏角、轴线对准奖励与六自由度 PPO

代码目录：`/data/nju/Lincao/InsertAnything_tactile`。更新日期：2026-10-13。
2026-10-12 将 `InsertAnythingTactile6D-*` 的姿态初始化与原版在手偏角实验统一。原版 InsertAnything 和 UniVTAC 源码本轮未修改；旧触觉任务、数据及 checkpoint 保留。

## 1. 新任务与坐标约定

新增 `InsertAnythingTactile6D-{物体}-Direct-v0`，物体可选：

| 物体 | 完整任务名 |
| --- | --- |
| 圆柱 | `InsertAnythingTactile6D-CircleHole-I-Direct-v0` |
| 方形 | `InsertAnythingTactile6D-SquareHole-II-Direct-v0` |
| L 形 | `InsertAnythingTactile6D-LHole-III-Direct-v0` |
| 三角形 | `InsertAnythingTactile6D-TriangleHole-IV-Direct-v0` |
| 六边形 III | `InsertAnythingTactile6D-HexagonHole-III-Direct-v0` |
| 六边形 IV | `InsertAnythingTactile6D-HexagonHole-IV-Direct-v0` |

TCP 为真实刚体 `panda_fingertip_centered`，四元数顺序 wxyz，主动旋转、列向量。
偏转轴 `n_T=[0,1,0]`，零偏角物体姿态 `R_TO(0)=RotY(pi)`，物体插入轴 `a_O0_T=[0,0,-1]`。
物体轴与孔轴均使用各自局部 +Z 的**同向**定义。物体真正插入时平移可以朝世界 −Z，这不改变两条轴的正方向约定。

冻结角度网络只输入左右原生触觉 RGB 和实际两指关节位移之和，输出一个带符号角度，单位度。奖励计算前转换为弧度：

```text
a_O_est_W = R_WT @ RotY(theta_est) @ [0, 0, -1]
a_H_W     = R_WH @ [0, 0, 1]
alpha     = acos(clamp(dot(a_O_est_W, a_H_W), -1, 1))
r_align   = weight * exp(-(alpha / radians(sigma_deg))**2) * contact_valid
```

不对点积取绝对值，否则倒置的插销也会被当作对齐。`R_WT` 来自机器人实际 TCP 姿态；`R_WH` 来自已知孔位姿。
TCP 位置不参与上述轴线旋转公式。

## 2. 控制、观测与奖励变化

- 六维动作是世界坐标下的 `[dx,dy,dz,delta_rx,delta_ry,delta_rz]`。后三维是旋转向量增量，左乘当前 TCP 四元数；不再把 roll 写成 π、pitch 写成 0。
- 保留原笛卡尔力矩控制器、动作 EMA、位置活动范围和自动闭爪；控制雅可比改用实际 TCP 刚体，与位姿误差的参考点一致。平移比例改为 0.003 m，旋转向量模长上限 0.035 rad。这是原控制结构的**每次物理子步目标偏移**，不是绝对位姿，也不是一次控制周期的累计角度上限。物理 120 Hz、策略 15 Hz、decimation=8。
- 原 policy/critic 向量后追加 5 维：预测角/15、估计轴在孔坐标系的 3 维方向、双侧接触有效标记。L 形任务 policy=31、critic=78、action=6。
- 原位置、深度、非圆形插销 yaw 对称性、动作平滑及力惩罚保留。新任务统一采用解耦位置奖励，避免原 keypoint 分支额外引入真值倾角奖励。
- 新轴线奖励默认 `weight=1`、`sigma=5°`。深度奖励再乘以 `exp(-(alpha/10°)^2)*contact_valid`。分阶段插入奖励还要求估计轴误差小于 3°，并限制物体不能穿到目标下方 2 mm 以外。
- 两侧各至少 64 个像素的压入深度大于 0.02 mm 才视为有效接触；接触无效时轴线和深度奖励归零，另加 −0.25 接触丢失项。接触标记不是网络置信度。
- 每个控制步使用同步渲染后的图像推理一次，奖励与观测共享结果；环境复位后重新推理。

新增轴线奖励不读取物体旋转真值。原奖励仍使用仿真的物体位置和 yaw，原非对称 critic 仍有物体真值状态；因此整个训练不是纯触觉、无真值监督的 RL。
真值偏角、真值轴误差、单轴外残差单独记录在 `tactile/gt_*`，用于诊断。

**当前初始化：夹爪竖直向下，插销产生 ±10° 在手偏角。** TCP 初始 roll=π、pitch=0，取消初始 roll/pitch 扰动，yaw 沿用各任务的原始分布。插销以 TCP 为支点，绕 TCP +Y（夹持接触面的法向）均匀采样 `theta ~ U(-10°, +10°)`；世界坐标中的转轴为 `R_WT @ [0,1,0]`。随后物理闭爪，实际姿态可能有小幅变化。

这与[原版在手偏角设置](README_insertanything_inhand_6d.md)使用同一段上游物体偏转代码：

```text
Delta_R_W = R_WT @ Ry(theta) @ R_WT.T
R_WO_new  = Delta_R_W @ R_WO
p_WO_new  = p_WT + Delta_R_W @ (p_WO - p_WT)
```

触觉版本保留已标定的几何中心抓持位置、胶垫和传感器安装方式；统一的是初始姿态、扰动轴、支点与默认角度范围。它原有的动作步长、触觉奖励、接触门限等设置也保留，所以这里没有将两套任务的全部参数设为相同。新默认 `--initial_tilt_deg 10` 控制插销偏转，不再控制夹爪的初始倾斜。配置同步为 `initial_in_hand_tilt_deg=10.0`、`ctrl.lock_roll_pitch=false`。旧 `InsertAnythingTactile-*` 任务保持原有行为。
上游抓持复位会推进整个物理场景，因此本版明确使用**整批同步、超时复位**；不支持只重置某个环境。时间耗尽返回 truncation，不作为任务终止。

## 3. 动态角度模型与历史验证

本节保留此前“夹爪倾斜 ±15°、插销竖直”初始化下的数据和测试结果。当前仍使用同一个冻结角度模型；新初始化的实测见第 8 节，旧结果不作为本次验收值。

原静态 v4 checkpoint：`outputs/normal_angle_v4_model_20261012/best.pt`，其新种子静态测试 MAE 为 0.751°。
接入物理抓持后，首次 L 形 2 环境 × 40 步检查得到动态 MAE **7.251°**、P95 **10.527°**，所以不能直接把静态精度当成 RL 中的精度。
原始诊断保留在 `outputs/tactile_rl_6d_check_20261012/`。

新增动态数据按真实物理抓持记录两侧 RGB、实际开度、物体/TCP 世界位姿、相对位姿、时间、环境/抓持/物体编号、种子、接触有效性和单轴外残差。
每次抓持保存控制步 0、4、8、12 的 4 帧。只用接触有效、实际角度在 ±15°、单轴外残差 ≤3° 的帧训练。
按**独立抓持分组**、物体及角度区间分层划分训练/验证/测试。同一次抓持的帧不能跨组。

动态校正沿用左右共享 CNN → 特征拼接 → 融合实际开度 → MLP → 单角度；RL 时网络冻结。
动态模型是单轴近似，仍不能识别任意三维在手滑转。机器人可以执行六自由度运动，不等于触觉网络能观测三个独立相对旋转自由度。

实测产物（路径均相对于项目目录）：

- 动态训练母集：`outputs/dynamic_angle_rl_train_20261012/`，六物体各 80 次抓持 × 4 帧，共 1,920 帧；有效 1,876 帧、469 次抓持。
- 327/71/71 次独立抓持划为 train/validation/test，对应 1,308/284/284 帧。按验证集选择 checkpoint；内部测试 MAE **0.997°**、P95 **2.839°**。
- 新种子数据：`outputs/dynamic_angle_rl_fresh_20261012/`，96 次抓持、384 帧，383 帧有效；没有用于训练或 checkpoint 选择。
- 新种子评估：`outputs/dynamic_angle_rl_fresh_eval_20261012/`，MAE **1.241°**、P95 **2.921°**、最大误差 **6.731°**。图：`angle_validation.png`；逐样本预测：`external_predictions.npz`。
- 冻结角度模型：`outputs/dynamic_angle_rl_model_20261012/best.pt`，SHA256 `3159f71b6d0e4d84cbf3e59e120b6ecfd5f6fa53dca288ee11d637e83e6dd358`。
- 新控制器、2 环境 × 40 步 roll/pitch 检查：`outputs/tactile_rl_6d_calibrated_check_20261012/`，接触有效率 100%，动态角度 MAE **2.079°**、P95 **3.731°**，两轴实际运动、无接触奖励归零和两次整批复位通过。这与静态图像测试、独立抓持测试是不同测试条件。
- 4 环境、2 轮 PPO 冒烟训练：`logs/rl_games/InsertAnythingTactile6D/smoke_6d_20261012/`，完成参数更新并生成 `nn/InsertAnythingTactile6D.pth` 和最后一轮权重。只验证训练流程，未训练出插入策略。重新加载此策略，2 环境 × 40 步检查通过，在线角度 MAE 2.610°、P95 4.004°；4 个缩短回合插入成功 0/4，结果在 `outputs/tactile_rl_6d_policy_check_20261012/`。

新种子分物体 MAE（每种约 64 帧、16 次独立抓持）：

| 圆柱 | 方形 | L 形 | 三角形 | 六边形 III | 六边形 IV |
| --- | --- | --- | --- | --- | --- |
| 0.950° | 1.103° | 1.885° | 1.524° | 0.878° | 1.109° |

本版可用于启动 RL 实验。上述误差尤其是尾部误差不能解释为精密插入保证；完整插入成功率仍需长训练后的多种子闭环评估。


## 4. RL 训练命令

本机运行环境为 Isaac Sim 5.1 / Isaac Lab 2.3 / Python 3.11，沿用 `scripts/python.sh` 指向的已配置运行时。
相机由脚本自动开启；脚本默认 4 环境是小规模验证起点，不是并行上限。下文模板保留该默认值；第 9 节实际训练/评估采用 16 环境。扩大并行数应同时检查显存和采样吞吐。

```bash
cd /data/nju/Lincao/InsertAnything_tactile
./scripts/python.sh scripts/train_tactile_rl.py \
  --task InsertAnythingTactile6D-LHole-III-Direct-v0 \
  --angle_checkpoint outputs/dynamic_angle_rl_model_20261012/best.pt \
  --num_envs 4 --headless \
  --max_iterations 1000 \
  --align_weight 1.0 --align_sigma_deg 5.0 \
  --initial_tilt_deg 10 --seed 0
```

`--angle_checkpoint` 是冻结的触觉角度网络；`--checkpoint` 才是恢复训练的 RL 策略权重。
新观测维度、控制方式和奖励均已改变，不能把官方旧 PPO checkpoint 当作本任务的可直接恢复权重。
当前 `run.json` 使用 `insertanything_tactile6d_rl_v2`，记录 `initialization`（模式、半范围、TCP 法向、旋转支点和抓持定位）。恢复训练和策略检查会验证任务、维度、初始化及角度模型 SHA256。旧 v1 的“夹爪倾斜”策略仍保留，但会被本入口拒绝直接续训/评估，避免混用初始化协议。当前新配置应从零训练新的 RL 策略。
更换物体时只替换完整任务名，每种物体分别训练策略。

先跑短训练可以使用：

```bash
./scripts/python.sh scripts/train_tactile_rl.py \
  --task InsertAnythingTactile6D-LHole-III-Direct-v0 \
  --angle_checkpoint outputs/dynamic_angle_rl_model_20261012/best.pt \
  --num_envs 4 --headless \
  --max_iterations 2 --horizon 16 --seq_length 8 --episode_steps 20
```

输出：`logs/rl_games/InsertAnythingTactile6D/<自动生成的运行名>/`。
`run.json` 绑定任务、初始化模式/范围和角度模型；`params/` 保存配置；`nn/` 保存策略；`summaries/` 保存 TensorBoard。

```bash
./scripts/python.sh -m tensorboard.main \
  --logdir logs/rl_games/InsertAnythingTactile6D --port 6006
```

重点检查触觉有效比例、预测角、估计/真值轴误差、真值偏角误差、单轴外残差、奖励及成功率。
训练迭代 2 次仅验证更新、保存与加载流程；不能据此声称策略已学会插入。

## 5. 控制检查与有限策略评估

检查 roll/pitch、动态推理、无接触奖励与同步复位：

```bash
./scripts/python.sh scripts/check_tactile_rl.py \
  --task InsertAnythingTactile6D-LHole-III-Direct-v0 \
  --angle_checkpoint outputs/dynamic_angle_rl_model_20261012/best.pt \
  --num_envs 2 --steps 40 --episode_steps 20 --seed 31 --headless
```

脚本输出 `report.json`、逐环境 `diagnostics.csv` 及复位/结束时双侧 PNG；默认执行有界 roll/pitch 探测动作。`report.json/reset_poses` 记录每次物理闭爪后的 TCP 倾角、采样/实际在手角度、孔轴夹角、四元数、预测值和接触有效性，并检查夹爪保持竖直。
CSV 的 `diagnostic_phase=post_action_pre_reset`：角度和成功诊断与动作奖励同属复位前状态；`post_reset_observation=true` 说明 Gym 返回的观测已复位。
评估新策略时，额外传入 `--policy_checkpoint` 加该次训练 `nn/` 下真实存在的 `.pth` 路径；脚本会验证与角度模型的绑定并执行确定性策略。
评估完整回合应移除短回合覆盖设置并设足够步数；本脚本的短检查不替代多种子插入成功率统计。

## 6. 动态角度数据和校正命令

以下模板采集的是当前“夹爪竖直、插销倾斜”的新初始化，数据 metadata 记录其 `initialization`。历史数据不被改写；旧采集源码备份在 `validation/upright_init_previous_sources_20261012/`。输出路径必须新建，不要覆盖已完成的实验目录。

```bash
./scripts/python.sh scripts/collect_dynamic_angle.py \
  --task InsertAnythingTactile6D-LHole-III-Direct-v0 \
  --num_envs 4 --grasp_batches 20 --seed 210000 --headless --initial_tilt_deg 10 \
  --output outputs/dynamic_angle_new/LHole-III

./scripts/python.sh scripts/adapt_dynamic_angle.py \
  --data outputs/dynamic_angle_new \
  --checkpoint outputs/normal_angle_v4_model_20261012/best.pt \
  --output outputs/dynamic_angle_new_model \
  --epochs 60 --device cuda:0
```

收齐六物体时对表中的六个任务分别执行采集，各用不同 seed，再把共同父目录传给 `--data`。
外部新种子测试使用独立输出目录采集，再运行：

```bash
./scripts/python.sh scripts/adapt_dynamic_angle.py \
  --evaluate_only --data outputs/dynamic_angle_new_fresh \
  --checkpoint outputs/dynamic_angle_new_model/best.pt \
  --output outputs/dynamic_angle_new_fresh_eval --device cuda:0
```

此示例需要先创建 `dynamic_angle_new_fresh` 新种子数据；评估会拒绝重复采集 UUID 或相同仿真 seed。

## 7. 配置与实现位置

均相对于 `/data/nju/Lincao/InsertAnything_tactile`：

| 文件 | 用途 |
| --- | --- |
| `source/insertanything_tactile/rl/config.py` | 六任务配置、权重路径、奖励宽度/权重、接触阈值、初始倾角范围 |
| `source/insertanything_tactile/rl/env.py` | 六自由度执行、同步推理、观测、奖励与诊断 |
| `source/insertanything_tactile/rl/agent.py` | 从原 InsertAnything PPO/LSTM 配置派生的相机批量配置 |
| `source/insertanything_tactile/rl/__init__.py` | 六个新 Gym 任务的显式注册 |
| `source/tactile_pose/rl_math.py` | 坐标转换、旋转增量、角度与高斯奖励 |
| `source/tactile_pose/rl_estimator.py` | 冻结模型加载、分侧标定检查、与离线一致的 RGB 预处理 |
| `scripts/train_tactile_rl.py` | 训练和兼容的策略恢复入口 |
| `scripts/check_tactile_rl.py` | 短控制检查、有限策略加载评估、诊断导出 |
| `scripts/collect_dynamic_angle.py` | 实际物理抓持数据采集 |
| `scripts/adapt_dynamic_angle.py` | 按抓持分组的角度校正和外部测试 |

单元测试与历史隔离清单检查：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./scripts/python.sh -m pytest -q -p no:cacheprovider tests
./scripts/python.sh scripts/verify_isolation.py
```

`verify_isolation.py` 对比的是最早接入触觉时的清单。此前用户已授权为原版新增在手偏角和六自由度开关，因此它会报告原版 `insertanything_env.py` 与 `insertanything_env_cfg.py` 的历史变化；本次没有重置那个历史清单。当前改动范围另行记录在 `validation/tactile_upright_init_20261012.json`。

## 8. 竖直夹爪初始化的本次验证（2026-10-12）

当前六个 `InsertAnythingTactile6D` 任务均已改为竖直 TCP、插销绕夹持法向 ±10°。各任务使用 seed 81，4 环境 × 40 控制步、20 步短回合；各测量 3 次物理闭爪后的姿态，共 72 次初始化。所有初始化的 TCP 竖直偏差均小于 0.072°，三组重置的在手偏角正常保留。

| 任务 | 最大 TCP 竖直偏差 | 闭爪最大偏角变化 | 动态预测 MAE | 动态预测 P95 |
| --- | --- | --- | --- | --- |
| CircleHole-I | 0.071° | 0.332° | 0.853° | 2.812° |
| HexagonHole-III | 0.066° | 1.157° | 1.214° | 2.059° |
| HexagonHole-IV | 0.069° | 2.311° | 0.528° | 1.452° |
| LHole-III | 0.063° | 0.730° | 1.831° | 4.375° |
| SquareHole-II | 0.069° | 1.329° | 1.215° | 2.339° |
| TriangleHole-IV | 0.071° | 0.798° | 0.539° | 1.231° |

上述六场景短运动检查中，双侧接触有效率均为 100%；合计 960 帧在线记录的预测 MAE 为 1.030°。这些帧来自少量相关轨迹，属于新初始化的集成检查，不代表独立大规模测试集上的精度。物理闭爪可改变采样偏角，报告始终分别记录采样值与实际值。

24 项已有单元测试通过。使用新初始化另完成 4 环境、2 轮 PPO 短训练，checkpoint epoch=2，优化器发生更新，所有保存的参数和优化器 tensor 均为有限值。新 v2 策略用 seed 82 重载检查通过：8 个缩短回合，估计/真值插入成功均为 0/8，触觉角度 MAE 为 1.007°。没有启动完整训练，也没有将短训练策略视为已学会插入。

本次产物：

- 六任务姿态、CSV、左右 RGB：`outputs/tactile_upright_init_20261012/<任务后缀>/`。
- 新策略回放：`outputs/tactile_upright_init_20261012/policy_reload/`。
- 物理检查汇总：`outputs/tactile_upright_init_20261012/physical_summary.json`。
- 新短训练策略：`logs/rl_games/InsertAnythingTactile6D/smoke_upright_inhand10_20261012/nn/InsertAnythingTactile6D.pth`。
- 修改范围与源码哈希：`validation/tactile_upright_init_20261012.json`。
- 之前的初始化源码备份：`validation/upright_init_previous_sources_20261012/`。

本次复用原版已经修改好的偏角采样代码，原版代码/资产/权重本轮保持不变；触觉奖励、观测拼接、六自由度执行函数与冻结角度权重也保持不变。


## 9. 2026-10-13：1000 轮训练后的完整评估

训练目录：`logs/rl_games/InsertAnythingTactile6D/LHole_III_tactile_inhand10_6d_20261012_233627`。
训练已完成：seed=42，16 环境，horizon=64，1000 轮，实际 frame=1,024,000。
最后权重为 `nn/last_InsertAnythingTactile6D_ep_1000_rew__96.08148_.pth`；`nn/InsertAnythingTactile6D.pth` 是最佳训练奖励权重，但 checkpoint 内 epoch=5，并不是第 1000 轮。

新增完整评估入口 `scripts/eval_tactile_rl.py`。它恢复训练保存的环境配置并验证 PPO 结构、输入归一化、初始化及冻结角度模型 SHA256；保持每回合 20 秒、300 个控制步。确定性评估各使用 16 环境 × 4 个新种子（1001/1002/1003/1004），两份权重的随机初始化配对记录在 `paired_reset_differences.json`。另以 seed=1005、16 环境检查随机动作执行。

| 权重及模式 | 完整回合 | 真值插入成功 | 双侧接触有效率 | 轨迹角度 MAE | 轨迹角度 P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 第 1000 轮，确定性 | 64 | 0/64 | 100.00% | 1.622° | 4.556° |
| 最佳奖励第 5 轮，确定性 | 64 | 0/64 | 100.00% | 1.589° | 4.417° |
| 第 1000 轮，随机动作 | 16 | 0/16 | 100.00% | 2.611° | 4.873° |

真值成功条件：原位置/深度/yaw 条件，额外要求深度误差 ≥−2 mm、真实轴线夹角 <3° 和双侧有效接触。报告同时记录由预测角计算的成功、去掉接触条件的几何成功、末帧成功以及连续至少 10 步成功；没有通过过滤失败初始化提高成功率。

第 1000 轮确定性动作在 19,200 个环境步中几乎固定，Z 动作始终为 +1。末帧平均 XY 误差 42.13 mm、深度误差 56.72 mm、真实轴线误差 39.78°。在手角度仍全部位于模型 ±15° 范围内。当前记录首先显示策略动作决策退化，不能把失败直接归因于无触觉接触。单角度估计的尾部误差仍需结合精密插入要求继续检查。

训练曲线也未显示稳定改善：最早最佳平均回报约 175.22，最后约 96.08；自适应学习率从 0.0001 最高升至 0.01。建议优先检查学习率上限、动作饱和和 PPO 更新稳定性。本次评估没有重新训练或修改控制/奖励。

可复现命令：

```bash
cd /data/nju/Lincao/InsertAnything_tactile
./scripts/python.sh scripts/eval_tactile_rl.py \
  --policy_checkpoint logs/rl_games/InsertAnythingTactile6D/LHole_III_tactile_inhand10_6d_20261012_233627/nn/last_InsertAnythingTactile6D_ep_1000_rew__96.08148_.pth \
  --num_envs 16 --seeds 1001 1002 1003 1004 --headless \
  --output "outputs/tactile_full_eval_$(date +%Y%m%d_%H%M%S)"
```

测试最佳奖励权重时将 checkpoint 文件名换成 `InsertAnythingTactile6D.pth`；随机动作检查加 `--stochastic`，可用 `--seeds 1005`。输出目录必须新建。

本次结果：`/data/nju/Lincao/InsertAnything_tactile/outputs/rl_eval_LHole_III_run20261012_233627_20261013`。
`comparison.json` 是三组汇总；`evaluation_diagnostics.png` 是闭环曲线；`training_curve.png` 和 `training_scalars.json` 是训练记录。每组 `report.json` 保存权重哈希和逐回合结果；每个种子目录保存 `diagnostics.csv`、`states.npz`、初始化位姿以及 env 0 的左右触觉 PNG（复位、1/150/300 步）。末帧状态和 RGB 均在自动复位前抓取；`raw_actions` 指策略限幅后、EMA 前动作，不是未限幅网络均值。

同期[原版中途权重评估](README_insertanything_inhand_6d.md#8-2026-10-13训练中的第-846-轮策略评估)使用 epoch=846、13,860,864 条训练转移，约为这次触觉训练的 13.5 倍。两版控制、奖励及夹持几何也有差异，当前数字不能单独归结为触觉传感器带来的效果。


## 10. 2026-10-13：偏角增大与训练退化的进一步诊断

详见[独立诊断报告](README_insertanything_tactile_failure_20261013.md)。已确认第 1000 轮策略在实际轨迹上几乎输出常量；真实触觉反馈配合比例纠偏能减小轴线夹角。日志与 checkpoint 检查显示学习率过度增长、较大 KL 和网络特征退化；奖励的窄高斯与多重门控也使大偏角区反馈很弱。本次仅诊断，未改写奖励或重新训练；修复后的成功率仍需新实验验证。
