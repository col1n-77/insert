# InsertAnything：六自由度插入、角度奖励与触觉扩展

基于 [mzhsoul/InsertAnything](https://github.com/mzhsoul/InsertAnything) 整理的研究代码，包含截至 **2026-10-15** 的本地修改。保留上游提交和 [BSD-3-Clause 许可证](LICENSE)，原始说明见 [README.upstream.md](README.upstream.md)。

| 代码位置 | 内容 | 任务入口 |
| --- | --- | --- |
| `source/InsertAnything`、`scripts/` | 原版任务；Isaac Sim 5.1 兼容；初始在手偏角；可选六自由度控制；完整回合评估 | `InsertAnything-*` |
| `InsertAnything_angle/` | 无触觉在手角度观测、轴线惩罚、奖励消融、可选深度倾角门控 | `InsertAnythingAngle-*` |
| `InsertAnything_tactile/` | 双侧 GelSight Mini / Taxim、采集与校准、姿态和单角度网络、动态角度校正、触觉六自由度 PPO | `InsertAnythingTactile-*`、`InsertAnythingTactile6D-*` |

三个系列保留独立入口，模型需要匹配各自的观测和配置。原版默认在手偏角为 0°、锁定 roll/pitch；Angle 默认随机在手偏角 ±10°、开放六自由度、角度观测噪声为 0°。新增深度倾角门控默认关闭。

## 安装和运行

原实验使用 Python 3.11、Isaac Sim 5.1、Isaac Lab 2.3.0、PyTorch 2.7.0+cu128 和 RL-Games 1.6.1。上游验证组合为 Isaac Sim 5.0 / Isaac Lab 2.2.1。先按各自官方说明安装仿真环境，再填写该环境 Python 的绝对路径：

```bash
export INSERTANYTHING_BASE_PYTHON=/absolute/path/to/isaaclab-env/bin/python
bash scripts/setup_local_env.sh
./scripts/python.sh scripts/list_envs.py
```

也可以直接指定已有完整运行环境，启动脚本会优先导入本仓库源码：

```bash
export INSERTANYTHING_PYTHON=/absolute/path/to/isaaclab-env/bin/python
./scripts/python.sh scripts/list_envs.py
./InsertAnything_angle/scripts/python.sh scripts/train_angle_rl.py --help
```

完整命令见[统一运行说明](docs/WORKFLOWS.md)，历史实验见[运行手册归档](docs/runbooks/README.md)，整理内容见[修改说明](docs/CHANGES.md)。历史手册中的本机绝对路径和结果链接指向原实验目录。

## CPU 验证

已有测试覆盖角度几何、噪声隔离、奖励和门控边界、光学校准、数据划分、姿态网络及在线角度估计：

```bash
python3 -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r requirements-test.txt
bash scripts/test_cpu.sh -q
```

已配置仿真环境时，可指定其 Python 直接测试，无需重新安装依赖。`requirements-test.txt` 和 GitHub Actions 只用于 CPU 测试，不替代 GPU 仿真或插入成功率评估。

## 资产、模型和实验结果

仓库保留上游孔与插销 USD。Isaac Sim 的机器人/场景资产需要安装环境、可用缓存或联网访问。触觉部分额外依赖独立安装的 TacEx、TacEx assets 和 Taxim 标定；已有生成器会在首次启动时生成本机 robot USD 覆盖层，见[资产说明](InsertAnything_tactile/assets/README.md)。

模型、数据、日志、视频、虚拟环境和生成的触觉 USD 层不纳入 Git。评估需要匹配的权重及训练目录中的 `params/env.yaml`、`params/agent.yaml`、`run.json`（对应扩展）。上游权重入口见 [checkpoint README](source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/README.md)。

历史对比中，角度奖励版与原版的观测和训练预算不同，不能当作只改变奖励的严格消融。当前源码含后续可选门控，旧模型成绩不代表门控版本已训练成功。各文件版权声明保留，外部 Isaac Sim、TacEx 及其资产遵循各自许可证。
