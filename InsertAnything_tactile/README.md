# InsertAnything_tactile

独立的 GelSight Mini 触觉扩展，复用仓库根目录 InsertAnything 的任务实现与外部权重，以及 UniVTAC/TacEx 的传感器资产和 Taxim 光学标定。原版与触觉扩展分别提供任务入口；触觉六自由度任务的初始化已与原版在手偏角实验统一。

当前可移植命令、依赖、数据字段与模型流程见[运行说明](../docs/WORKFLOWS.md)。

历史配置与实测记录见：
[InsertAnything 触觉扩展运行手册](../docs/runbooks/README_insertanything_tactile.md)。

双侧触觉＋实测夹爪开度预测物体相对旋转：
[姿态数据采集、第一版网络、训练与推理手册](../docs/runbooks/README_insertanything_tactile_pose.md)。

[全部六场景与同步双侧触觉图集、截图重现命令](../docs/runbooks/README_insertanything_tactile_scenes.md)。

本扩展使用 PhysX 刚体胶垫与 Taxim 深度渲染，不包含 UniVTAC 的 UIPC 胶体形变和标记剪切模型。旧 `InsertAnythingTactile-*` 的触觉输出作为独立数据接口提供；`InsertAnythingTactile6D-*` 使用角度预测参与观测和奖励。

代码入口：`source/insertanything_tactile/`；运行脚本：`scripts/`；本机生成的 USD 覆盖层：`assets/`；运行产物写入 `outputs/`。历史 `validation/` 产物未上传。

参考实现（仅引用）：

- `/data/nju/Lincao/InsertAnything/source/InsertAnything/InsertAnything/tasks/direct/insertanything/`
- `/data/nju/Lincao/UniVTAC/envs/sensors/tactile.py`
- `/data/nju/Lincao/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/sensors/gelsight_mini/gsmini_taxim.py`
- `/data/nju/Lincao/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/robots/franka/franka_gsmini_gripper_rigid.py`

当前分侧校准、600 次抓取重新训练及独立角度测试：[v2 运行手册](../docs/runbooks/README_insertanything_tactile_pose_v2.md)。

新的接触姿态数据生成入口：[v3 六物体 TCP 采样、随机压入、双侧 RGB 与实际位姿](../docs/runbooks/README_insertanything_tactile_tcp_v3.md)。已验收六物体各 10 组有效样本；旧 v2 模型/数据继续保留。

静态采集与预测入口：[v4 竖直插销、夹持法向在手偏角、单输出网络](../docs/runbooks/README_insertanything_tactile_normal_v4.md)。已生成 600 组母集＋120 组独立测试数据；仅回归一个角度，独立测试 MAE 0.751°。

六自由度触觉 RL：[轴线对准奖励、动态角度校正、六任务 PPO 训练与评估命令](../docs/runbooks/README_insertanything_tactile_rl_6d.md)。使用独立 `InsertAnythingTactile6D` 任务；当前初始化为夹爪竖直向下、插销绕 TCP Y 轴随机偏转 ±10°。新旧初始化的实测与训练/加载命令见手册；历史 1000 轮策略已评估，出现明显退化，不能认为触觉策略已学会插入。详见[失败诊断](../docs/runbooks/README_insertanything_tactile_failure_20261013.md)。

跨机器安装与当前命令见[统一运行说明](../docs/WORKFLOWS.md)。上述历史手册中的绝对路径和结果目录仅表示原实验记录。
