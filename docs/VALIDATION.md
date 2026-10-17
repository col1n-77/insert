# 发布验证：2026-10-17

本次在独立发布副本中完成检查；运行中的工作站训练源码哈希未变化。没有为发布另行启动 GPU 仿真、数据采集或长训练。

## 已执行

- CPU 回归：**42 passed, 8 subtests passed**，使用已有 Python 3.11 / PyTorch 2.7.0 环境，禁用 CUDA 和外部 pytest 插件自动加载。
- 新增队列检查覆盖仓库根目录资产/源码定位、两组命令只差权重和名称、plan-only 不启动训练、错误 KL 阈值拒绝及显式 constant 兼容。
- 在实际发布布局中运行 `run_matched_reward_study.py --plan_only`，确认原版任务文件进入源码哈希清单，未启动仿真。
- 101 个 Python 文件 AST 语法通过、5 个 Shell 文件语法通过、严格 JSON 配置解析通过。
- README、当前运行文档与模块入口的 49 个仓库相对文件链接通过；24 个 Bash 代码块语法通过，25 条脚本命令的参数与本地 parser 核对通过。
- 自适应训练、动态采集、触觉评估入口的命令行帮助检查通过。动态采集入口受 Isaac AppLauncher 的预解析影响，查看帮助时同时提供 `--output /tmp/unused_help`。
- 发布快照逐文件 SHA-256、Git 空白检查与提交范围检查在提交前完成。最终原工作站采集/模型产物未纳入 Git。

这些检查不代表在另一台机器上已经完成 Isaac/TacEx 安装、GPU 触觉渲染或完整插入成功率验证。跨机器使用前仍需准备 README 中列出的仿真组件、资产和匹配模型。

## 本地或 CI 执行

已有环境：

```bash
export INSERTANYTHING_PYTHON=/absolute/path/to/python
bash scripts/test_cpu.sh -q
python3 InsertAnything_tactile/scripts/verify_isolation.py
```

独立 CPU 测试环境（Python 3.11）：

```bash
python3 -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r requirements-test.txt
bash scripts/test_cpu.sh -q
```

`requirements-test.txt` 只覆盖 CPU 测试，不替代完整仿真依赖。GitHub Actions 配置在 `.github/workflows/cpu-tests.yml`；远端执行结果以仓库 Actions 为准。

`verify_isolation.py` 核对的是此发布版本的相对路径快照，主动修改被记录文件后出现差异是预期现象；它不核验外部 TacEx/Isaac 环境。

2026-10-15 的首次发布验证为 37 项测试、8 项子测试；本次新增 5 个队列用例。历史运行和物理评估证据见 [runbooks](runbooks/README.md)。
