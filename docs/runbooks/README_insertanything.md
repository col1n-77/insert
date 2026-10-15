# InsertAnything：Isaac Sim 5.1 环境、训练与官方权重验证

更新：2026-10-09。统一索引：[README.md](README.md)。

项目代码：`/data/nju/Lincao/InsertAnything`。本页是本机当前运行入口；上游说明仍保留在项目仓库。

2026-10-12 新增可选的[原版在手偏角与六自由度训练](README_insertanything_inhand_6d.md)：不添加视触觉传感器，原任务奖励、观测、资产和默认控制行为保留；通过命令行开启插销偏角 ±10° 与 roll/pitch 动作。下文官方权重成绩仍对应原始默认设置。

## 1. 当前结论与版本

本机可以运行 InsertAnything 的仿真、RL-Games PPO 训练和官方权重评估。使用已经安装的 Isaac Sim 5.1 与 Isaac Lab 2.3.0，不需要将现有仿真环境降级。

[项目上游](https://github.com/mzhsoul/InsertAnything)验证的是 Isaac Sim 5.0、Isaac Lab 2.2.1 和 Python 3.11。本机采用较新的组合；[Isaac Lab 2.3.0 官方发布说明](https://github.com/isaac-sim/IsaacLab/releases/tag/v2.3.0)明确对应 Isaac Sim 5.1。本页的运行验证不能代替对论文软件版本和实验协议的精确复现。

| 项目 | 本机实际配置 |
| --- | --- |
| OS | Ubuntu 24.04.4 LTS |
| CPU / RAM | Intel Core Ultra 9 285K / 约 46 GiB |
| GPU | RTX 5090 D，32 GB 显存 |
| NVIDIA 驱动 | 580.178.04 |
| Python | 3.11.16 |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab 核心 | 2.3.0，来自已安装的官方 Python 包 |
| Isaac Lab RL / Tasks 安装版本 | 0.5.0 / 0.11.14 |
| PyTorch / CUDA runtime | 2.7.0+cu128 / 12.8 |
| RL-Games | 1.6.1，Isaac Sim 的 Python 3.11 分支 |
| Gymnasium / Warp | 1.2.0 / 1.17.0 |
| NumPy | 1.26.0 |

项目虚拟环境是 `/data/nju/Lincao/InsertAnything/.venv-isaac51`，通过 `--system-site-packages` 借用 `/data/nju/miniforge3/envs/UniVTAC-isaac51` 的仿真组件，仅将 InsertAnything 以 editable 模式安装到项目环境。没有修改基座 Conda 环境中的包。

Isaac Lab RL / Tasks 扩展借用基座已有的 editable 安装，源码指向 FlexiTac 仓库。因此更新这些外部源码或基座依赖后，应重新运行本页的检查。这里的虚拟环境没有独立复制整套仿真软件。

`nvidia-smi` 显示的 CUDA 13.0 是驱动支持上限；实际 PyTorch 使用 CUDA 12.8，两者不要求数值相同。GPU CUDA 运算和实际 PhysX 仿真均已验证。

## 2. 启动与重新安装

推荐使用项目启动脚本，不必先切换 Conda 环境：

```bash
cd /data/nju/Lincao/InsertAnything
./scripts/python.sh scripts/list_envs.py
```

也可以直接激活项目环境：

```bash
cd /data/nju/Lincao/InsertAnything
source .venv-isaac51/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
python -c 'import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))'
```

重新安装项目扩展并检查依赖：

```bash
cd /data/nju/Lincao/InsertAnything
bash scripts/setup_local_env.sh
./scripts/python.sh -m pip --no-cache-dir check
```

安装脚本检查 Python 3.11、所需的 Isaac Lab / RL-Games 包和既有虚拟环境的基座。缺少基座环境时会明确报错；脚本不会自动安装整套 Isaac Sim。

本次最终环境的依赖检查结果是 `No broken requirements found.`。早期 `env_isaaclab` 开发版基座检查记录保留在日志目录，原 `.venv` 是本次探索时创建的环境；当前命令统一使用 `.venv-isaac51`。

项目启动脚本开启无缓冲输出，避免仿真器快速退出时丢失 Python 日志。训练入口还修正了两处兼容问题：配置快照改用标准库 `pickle` 保存，日志警告改用 `carb.log_warn`。任务资产、奖励和控制参数保持上游实现。

## 3. 六个任务与有限仿真检查

| 任务 ID | 几何 / 公差 | Actor 观测维度 |
| --- | --- | ---: |
| `InsertAnything-CircleHole-I-Direct-v0` | 圆孔 / I | 23 |
| `InsertAnything-SquareHole-II-Direct-v0` | 方孔 / II | 23 |
| `InsertAnything-LHole-III-Direct-v0` | L 形 / III | 26 |
| `InsertAnything-TriangleHole-IV-Direct-v0` | 三角形 / IV | 23 |
| `InsertAnything-HexagonHole-III-Direct-v0` | 六边形 / III | 26 |
| `InsertAnything-HexagonHole-IV-Direct-v0` | 六边形 / IV | 23 |

六个任务均通过最终环境中的 4 个并行环境、100 步检查。检查包含资产生成、初始重置、动作推进以及观测、奖励和接触力的有限值验证；100 步不足以覆盖所有任务的完整回合。

单任务检查命令会自动结束：

```bash
cd /data/nju/Lincao/InsertAnything
./scripts/python.sh scripts/smoke_test.py \
  --task InsertAnything-CircleHole-I-Direct-v0 \
  --num_envs 4 --steps 100 --headless \
  --report logs/local_check/circle.json
```

本机首次启动已成功读取机器人和桌子 USD。项目内带有插销与孔的 USD；机器人、桌子等 NVIDIA 资产仍需要可用缓存或网络访问。

任务主要验证已经握持插销后的精密插入。L 形与六边形 III 的 actor 使用紧凑三轴力反馈，并非触觉相机图像输入。

## 4. PPO 训练

本次圆孔任务完成 4 个环境、1 次 PPO 迭代，保存了模型。进一步读取 checkpoint 确认模型张量全部有限、Adam 状态非空且已有优化步，说明训练确实更新了参数。

重复有限训练检查时使用新的实验名：

```bash
cd /data/nju/Lincao/InsertAnything
./scripts/python.sh scripts/rl_games/train.py \
  --task InsertAnything-CircleHole-I-Direct-v0 \
  --num_envs 4 --max_iterations 1 --headless --seed 0 \
  agent.params.config.full_experiment_name="local_check_$(date +%Y%m%d_%H%M%S)"
```

完整训练入口（本次没有执行长训练）：

```bash
cd /data/nju/Lincao/InsertAnything
./scripts/python.sh scripts/rl_games/train.py \
  --task InsertAnything-CircleHole-I-Direct-v0 \
  --num_envs 128 --headless --seed 0 \
  agent.params.config.full_experiment_name="circle_I_$(date +%Y%m%d_%H%M%S)"
```

上游配置默认训练 1,000 个迭代，`--max_iterations` 可覆盖上限。其他任务只替换任务 ID；维持任务和 checkpoint 对应。默认配置含 LSTM 和 privileged critic。

日志、配置快照和模型目录：

```text
/data/nju/Lincao/InsertAnything/logs/rl_games/InsertAnything/<实验名>/
  params/
  summaries/
  nn/
```

本次短训练产物位于 `setup_isaac51_20261009`。`last_InsertAnything_ep_1_rew_-inf.pth` 的 `-inf` 表示停止前还没有完成整回合，不能据此判断训练出的策略成功率；它仅用于训练流程验收。

查看训练曲线（选择当前未占用的端口）：

```bash
cd /data/nju/Lincao/InsertAnything
./scripts/python.sh -m tensorboard.main \
  --logdir logs/rl_games/InsertAnything --port 6006
```

打开 [TensorBoard](http://localhost:6006)。本次有限训练检查以模型与优化器状态为依据，不将短训练曲线作为学习效果证据。

## 5. 官方 checkpoint 与本机评估

两个文件已下载，均为 213,716,895 字节，SHA256 与上游 checkpoint README 一致：

| 文件 | SHA256 |
| --- | --- |
| `InsertAnything-LHole-III-Direct-v0.pth` | `fc158b0d72afdab0d860ceb379f8e3ec2b0b693d49c9b24629608a5bea4b58ac` |
| `InsertAnything-HexagonHole-III-Direct-v0.pth` | `d11d9a4849238dbdec4602efb86df6d260118fb7ca59e0f159a05546d2b14c08` |

位置：`/data/nju/Lincao/InsertAnything/source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/`。

本次各执行 1 个完整批次，seed 0、128 个环境，保留任务配置的随机策略模式 `player.deterministic=False`：

| 任务 | 成功数 | 成功率 | 成功试验的平均成功时间 |
| --- | ---: | ---: | ---: |
| L 形 III | 122/128 | 95.31% | 6.54 s |
| 六边形 III | 122/128 | 95.31% | 5.57 s |

这些结果证明当前组合可以加载官方策略并完成批量插入。上游报告使用 Isaac Sim 5.0、每个任务 3 批次，本次仅 1 批次；不能将本次数字当作论文完整复现结果。

本次实测命令：

```bash
cd /data/nju/Lincao/InsertAnything
./scripts/python.sh scripts/rl_games/play.py \
  --task InsertAnything-LHole-III-Direct-v0 \
  --num_envs 128 --headless --seed 0 --max_episodes 1 \
  --checkpoint source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/InsertAnything-LHole-III-Direct-v0.pth

./scripts/python.sh scripts/rl_games/play.py \
  --task InsertAnything-HexagonHole-III-Direct-v0 \
  --num_envs 128 --headless --seed 0 --max_episodes 1 \
  --checkpoint source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/InsertAnything-HexagonHole-III-Direct-v0.pth
```

需要对照上游批次数时，将 `--max_episodes 1` 改为 `3`，即每个任务 384 次试验；仿真版本仍与上游不同。`max_episodes` 统计同步环境批次，不是单个环境试验总数。

GUI 查看命令（参数已对照脚本，窗口播放未作为本次验收项）：

```bash
cd /data/nju/Lincao/InsertAnything
./scripts/python.sh scripts/rl_games/play.py \
  --task InsertAnything-LHole-III-Direct-v0 \
  --num_envs 1 --seed 0 --max_episodes 1 --real-time \
  --checkpoint source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/InsertAnything-LHole-III-Direct-v0.pth
```

需要可用的图形显示会话；省略 `--max_episodes` 会持续播放。录像可加 `--video --video_length 300 --enable_cameras`，本次没有执行录像。官方权重的录像默认写到任务源码目录下的 `videos/play/`。

## 6. 力信号记录与已有结果查看

力 CSV 仅在 evaluation 模式、接触力启用、显式开启记录且 `num_envs=1` 时写入。以下采集命令已核对参数，本次没有运行力记录采集：

```bash
cd /data/nju/Lincao/InsertAnything
./scripts/python.sh scripts/rl_games/play.py \
  --task InsertAnything-LHole-III-Direct-v0 \
  --num_envs 1 --headless --seed 0 --max_episodes 1 \
  --checkpoint source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/InsertAnything-LHole-III-Direct-v0.pth \
  env.task.contact_force.log_contact_force=True
```

CSV 目录：`source/InsertAnything/InsertAnything/tasks/direct/insertanything/contact_force_logs/`。字段为 `step`、`is_engaged`、`is_engaged_half`、`is_success` 与 `contact_force_x/y/z`。重置预热也可能创建文件，应结合文件名和 step 内容识别评估回合。

先完成上述记录后再查看文件；没有记录时目录可能不存在：

```bash
cd /data/nju/Lincao/InsertAnything
ls -lt source/InsertAnything/InsertAnything/tasks/direct/insertanything/contact_force_logs/
```

本次验收的原始日志与 JSON 报告在：

```text
/data/nju/Lincao/InsertAnything/logs/setup_20261009/
  environment_isaac51.json
  pip_check_isaac51.log
  smoke_isaac51_*.json
  training_isaac51.json
  train_circle_isaac51_1epoch.log
  checkpoints.json
  evaluation_isaac51.json
  play_isaac51_LHole-III.log
  play_isaac51_HexagonHole-III.log
```

真实机器人部署另外需要兼容的 Franka 控制服务、已标定目标位姿及真实力传感器接口。本次完成的是仿真环境配置与运行验证，未配置或连接真实机器人控制链路。
