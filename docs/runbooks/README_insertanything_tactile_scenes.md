# InsertAnything + tactile：六场景与双侧触觉图集

更新：2026-10-12。图集按当前 `insertanything_tactile.TASKS` 注册表生成，包含全部 6 个任务。原 InsertAnything / UniVTAC 源码与任务配置保留；图像、原始数组和采样报告位于独立触觉项目的新输出目录。

## 查看图集

输出目录：[scene_gallery_20261012_all](/data/nju/Lincao/InsertAnything_tactile/outputs/scene_gallery_20261012_all/)。

- [场景与触觉图集 1：圆孔 I、方孔 II、L 孔 III](/data/nju/Lincao/InsertAnything_tactile/outputs/scene_gallery_20261012_all/scenes_and_tactile_1.png)
- [场景与触觉图集 2：三角孔 IV、六角孔 III、六角孔 IV](/data/nju/Lincao/InsertAnything_tactile/outputs/scene_gallery_20261012_all/scenes_and_tactile_2.png)
- [六场景机械臂全景](/data/nju/Lincao/InsertAnything_tactile/outputs/scene_gallery_20261012_all/all_scenes_overview.png)
- [HTML 图集：筛选场景、点击查看原始分辨率](/data/nju/Lincao/InsertAnything_tactile/outputs/scene_gallery_20261012_all/index.html)
- [配置、接触和完整性报告](/data/nju/Lincao/InsertAnything_tactile/outputs/scene_gallery_20261012_all/gallery.json)

图集每行从左到右：**夹爪/插销/孔位近景、孔位近景、左触觉 RGB、右触觉 RGB**。机械臂全景单独提供；每个任务的文件夹也有原始全景、近景和双侧 RGB。

![圆孔、方孔、L 孔与双侧触觉](/data/nju/Lincao/InsertAnything_tactile/outputs/scene_gallery_20261012_all/scenes_and_tactile_1.png)

![三角孔、六角 III、六角 IV 与双侧触觉](/data/nju/Lincao/InsertAnything_tactile/outputs/scene_gallery_20261012_all/scenes_and_tactile_2.png)

## 场景与独立图片

完整任务 ID 为 `InsertAnythingTactile-<下表后缀>-Direct-v0`。尺寸是任务配置中 `diameter` 字段的标称值，非额外几何测量。

| 场景 | 任务后缀 | 插销/孔标称尺寸（mm） | 单场景组合图 |
|---|---|---:|---|
| 圆孔 I | `CircleHole-I` | 8.000 / 10.000 | [圆孔 I](/data/nju/Lincao/InsertAnything_tactile/outputs/scene_gallery_20261012_all/CircleHole-I/scene_and_tactile.png) |
| 方孔 II | `SquareHole-II` | 9.500 / 10.000 | [方孔 II](/data/nju/Lincao/InsertAnything_tactile/outputs/scene_gallery_20261012_all/SquareHole-II/scene_and_tactile.png) |
| L 孔 III | `LHole-III` | 14.900 / 15.000 | [L 孔 III](/data/nju/Lincao/InsertAnything_tactile/outputs/scene_gallery_20261012_all/LHole-III/scene_and_tactile.png) |
| 三角孔 IV | `TriangleHole-IV` | 11.960 / 12.000 | [三角孔 IV](/data/nju/Lincao/InsertAnything_tactile/outputs/scene_gallery_20261012_all/TriangleHole-IV/scene_and_tactile.png) |
| 六角孔 III | `HexagonHole-III` | 11.884 / 12.000 | [六角孔 III](/data/nju/Lincao/InsertAnything_tactile/outputs/scene_gallery_20261012_all/HexagonHole-III/scene_and_tactile.png) |
| 六角孔 IV | `HexagonHole-IV` | 11.976 / 12.000 | [六角孔 IV](/data/nju/Lincao/InsertAnything_tactile/outputs/scene_gallery_20261012_all/HexagonHole-IV/scene_and_tactile.png) |

六角 III / IV 共用孔资产，使用不同插销尺寸、初始化角度与奖励/旋转对称性配置，因此截图外观接近；它们是两个独立任务。

## 采样含义与原始文件

每个任务均使用 `num_envs=1`、seed 0，初始化并物理闭爪后保持零动作 8 个控制步，约 0.533 s。图中是该时刻的夹持状态。不同外部视角只改变展示相机，并未推进物理仿真；截图与左右触觉来自同一个物理状态。本图集没有执行插入策略，不提供成功率结论。

传感器为已有 GelSight Mini / Taxim，胶垫接触采用 PhysX 刚体模型。触觉 RGB 是实际仿真输出，不作镜像、增强或替换。各侧接触判定：至少 64 个像素的几何压入超过 0.02 mm；压入深度不等于标定后的接触力或压力。

本次圆孔 I、方孔 II 的左侧图像按当时共用的 28 mm 模型零点未达到接触阈值。后续 [光学零点诊断](README_insertanything_tactile_symmetry.md) 确认左右胶垫安装深度相差约 0.295 mm；图中的“弱接触 / 未达阈值”标签受到这一系统偏差影响，不能直接作为实际未接触的结论。图像和数值按原配置保留，校准对照见诊断页。

每个场景子目录包含：

| 文件 | 内容 |
|---|---|
| `scene.png` | 机械臂全景，1280×960 |
| `detail.png` | 夹爪、传感器、插销与孔位近景，1280×960 |
| `hole.png` | 孔位近景，1280×960 |
| `left_tactile.png`、`right_tactile.png` | 左右原始 RGB，320×240 |
| `snapshot.npz` | 双侧 RGB、几何压入深度、相机深度及实际物体/孔位位姿、关节位置/速度 |
| `report.json` | 任务 ID、种子、时刻、展示相机参数、尺寸、资产路径、实际关节开度及接触状态 |
| `scene_and_tactile.png` | 单场景组合图 |

所有任务的五张原图检查为非空；双侧 PNG 与 NPZ 中的 RGB 数组逐像素一致；全部原始字段有限，接触像素统计与报告一致。

## 重现与离线重新排版

使用已配置的 Python / Isaac Sim 环境，不需额外模型权重。依赖与项目启动说明见 [触觉扩展运行手册](README_insertanything_tactile.md)。

全部场景重新渲染，默认自动创建新的时间戳目录：

```bash
cd /data/nju/Lincao/InsertAnything_tactile
./scripts/python.sh scripts/render_scene_gallery.py --headless --seed 0 --steps 8
```

各场景依次运行，避免同时占用 GPU；完成后生成两张总览图、六张单场景组合图、全景图、HTML 与汇总报告。可指定 `--output outputs/自定义新目录`，默认拒绝已有目录。

只重新排版已有截图，不启动 Isaac Sim：

```bash
cd /data/nju/Lincao/InsertAnything_tactile
./scripts/python.sh scripts/render_scene_gallery.py --assemble_only --output outputs/scene_gallery_20261012_all
```

单独渲染一个任务，例如方孔：

```bash
cd /data/nju/Lincao/InsertAnything_tactile
./scripts/python.sh scripts/render_scene.py --headless --task InsertAnythingTactile-SquareHole-II-Direct-v0 --steps 8 --seed 0 --output outputs/square_scene_new
```

脚本只改变进程内展示相机设置，不回写任务配置、机器人资产或物体资产。前期相机视角探测输出在 `outputs/scene_gallery_20261012/` 保留，正式图集为上面的 `scene_gallery_20261012_all/`。

2026-10-12 已完成[分侧校准并重新采集训练](README_insertanything_tactile_pose_v2.md)。本文图集保留旧基准；之后 `render_scene.py` 的默认运行使用校准基准，并在报告记录版本。
