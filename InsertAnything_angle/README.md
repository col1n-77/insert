# InsertAnything Angle：无触觉单角度观测与奖励消融

原机器人不添加触觉。LHole-III 的 Actor/Critic 在原 26/73 维后追加实际在手角度，变为 27/74 维；新增角度噪声默认为 0°。TCP 初始竖直向下，插销绕 TCP +Y 采样 ±10°，执行开放六自由度。

每步奖励为 `r_original + w * (exp(-(alpha/sigma)^2) - 1)`，其中 alpha 是真实物体/孔轴线夹角，对准时新增项为 0。可选 `--depth_axis_gate` 仅调整连续深度项，默认关闭。

- [完整安装、训练、评估与状态记录命令](../docs/WORKFLOWS.md)
- [同配置 0/1 奖励对照、预算与学习率](../docs/MATCHED_REWARD_STUDY.md)
- [历史实验、TCP 停滞和力分析](../docs/runbooks/README_insertanything_depth_axis_gate.md)

最新 `scripts/run_matched_reward_study.py` 将两组训练、冻结评估及汇总串联；默认 adaptive，初始学习率 1e-4，KL 阈值 0.008。`scripts/train_angle_rl.py` 为兼容历史命令默认仍为 constant，使用自适应时显式传 `--lr_schedule adaptive --kl_threshold 0.008`。Critic 保留原 rl_games 实际行为。

从仓库根目录查看计划（不会启动训练）：

```bash
./InsertAnything_angle/scripts/python.sh scripts/run_matched_reward_study.py \
  --study_name "plan_$(date +%Y%m%d_%H%M%S)" --plan_only
```

正式训练用新的实验名并去掉 `--plan_only`。运行脚本默认复用仓库根目录的 `.venv-isaac51`，也支持 `INSERTANYTHING_PYTHON`。当前匹配队列专用于 LHole-III，单次训练支持六个已注册任务。

本目录不包含训练数据和 checkpoint；`logs/`、`outputs/` 在运行后生成。源码沿用上游相应文件许可证。
