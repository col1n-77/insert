# InsertAnything Angle：无触觉单角度观测与奖励消融

基于本机修改过的 [InsertAnything](https://github.com/mzhsoul/InsertAnything) 新增独立任务包，不改写原项目。
复用仓库根目录 `.venv-isaac51` 运行环境、原任务代码与资产；也可通过环境变量指定 Python。首次物理检查与训练结果见运行手册。

当前操作文档统一位于：

- [环境、训练、奖励对照、评估及结果查看](../docs/runbooks/README_insertanything_angle_rl.md)
- [可选深度倾角门控：实现、验证与训练](../docs/runbooks/README_insertanything_depth_axis_gate.md)
- [全部操作文档索引](../docs/runbooks/README.md)

`source/insertanything_angle` 提供六个 `InsertAnythingAngle-*` 任务。
原机器人不添加触觉传感器；原观测后追加实际在手角度，当前默认不加角度观测噪声（标准差 0°）；每一步新增 `w*(exp(-(alpha/sigma)^2)-1)` 与原奖励相加，对准时新增项为 0。
2026-10-15 新增可选 `--depth_axis_gate`：用真实物体/孔轴夹角平滑门控连续深度项；默认关闭，旧模型奖励保持兼容。
默认夹爪初始竖直向下，物体绕 TCP Y 轴随机 ±10°，执行开放六自由度。

源程序沿用上游相应文件许可证；本目录为独立实验扩展，未修改上游训练权重。

跨机器安装与当前命令见[统一运行说明](../docs/WORKFLOWS.md)。下方历史手册中的绝对路径和结果目录仅表示原实验记录。
