# 2026-10-15 源码整理

上游基线为 `mzhsoul/InsertAnything` 的 `f033da5`，保留上游提交和原许可证。

## 收录的本地修改

- 原版训练入口兼容修复：标准库 pickle 快照、`carb.log_warn`。
- 可配置初始在手偏角和 roll/pitch 开关；原默认值保留。
- 原版有限仿真检查、六自由度检查及冻结权重完整回合评估。
- Angle 独立任务、角度观测、噪声隔离、轴线惩罚、严格成功诊断、可选连续深度门控、训练和奖励研究工具。
- Tactile 独立任务、双侧传感器、分侧校准、采集、姿态和单角度模型、动态适配、六自由度 PPO、既有 CPU 测试。
- 14 篇原工作站 InsertAnything 运行手册归档。

## 发布整理

- 在同一仓库中保留原版、`InsertAnything_angle/` 和 `InsertAnything_tactile/` 的独立入口。
- 调整扩展启动器及两处原版路径查找，默认复用仓库根目录；支持 `INSERTANYTHING_PYTHON`、`INSERTANYTHING_ROOT`。
- 启动时优先将当前仓库源码放入 PYTHONPATH，避免误用其他 editable 安装。
- 安装脚本使用用户指定或 PATH 中的 Python，移除固定工作站路径。
- 统一 README 与运行说明，保留上游原始 README 和历史记录。
- 过滤权重、数据、日志、环境与缓存。触觉 robot USD 覆盖层在使用环境中通过原有生成器创建。
- 旧本机隔离检查改为仓库相对路径的快照完整性检查。
- 为既有 CPU 测试增加统一脚本、依赖清单和 GitHub Actions。

奖励、控制器、传感器和网络的科学实现从本机源码复制，本次未改实验参数或启动长训练。来源及逐文件哈希见 [source_snapshot.json](source_snapshot.json)，发布时调整过的文件标记为 `packaging_changed`。

原工作站的虚拟环境、数据、模型、日志和视频保留原位。外部 TacEx 网格与标定需要独立安装；历史手册中的绝对路径和结果链接不代表这些产物包含在仓库中。
