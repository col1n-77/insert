# InsertAnything：左右触觉不对称与光学零点诊断

更新：2026-10-12。此页补充解释 [六场景图集](README_insertanything_tactile_scenes.md) 中系统性的“左浅、右深”。当前独立触觉任务已接入分侧零点；[v2 校准、采集、训练与实测结果](README_insertanything_tactile_pose_v2.md)是当前运行入口。下方保留旧 28 mm 模式的原因分析及诊断对照，原始图集和 v1 数据、模型均保留。

## 已确认的主要原因

当前借用的 TacEx Franka / GelSight Mini 刚体资产中，左右胶垫相对传感器相机的安装深度并不完全对称。按同一 USD 坐标系计算胶垫网格外表面到相机像平面的距离，分别为 **28.246235 mm（左）、27.950882 mm（右）**，相差 **0.295353 mm**。

原来的 `GelSightSensor._get_height_map()` 使用 `near_clip + gelpad_height = 24 + 4 = 28 mm` 作为左右共用基准；几何压入代理等于 `max(28 mm - camera_depth_mm, 0)`。因此左侧被低估约 0.246 mm，右侧被高估约 0.049 mm。这个固定误差足以使左侧显示近似背景、右侧出现明显接触边缘。

前期将圆孔、方孔左侧的“未达到模型接触阈值”直接解释为实际弱接触，判断过早。该标志受到未经分侧校准的光学零点影响，**不能直接证明左胶垫没有物理接触**。

## 方孔同一帧的对照

方孔 II、seed 0、初始化后保持 8 步。物体和夹爪姿态保持不变，只在内存中将原始相机深度按各侧胶垫几何零点重新送入同一个 Taxim 光学模型。

| 指标 | 左 | 右 |
|---|---:|---:|
| 旧共用零点（mm） | 28.000 | 28.000 |
| 胶垫几何零点（mm） | 28.246 | 27.951 |
| 旧最大几何压入代理（mm） | 0.000 | 0.317 |
| 分侧零点对照最大压入代理（mm） | 0.238 | 0.268 |
| 旧接触像素数，阈值 0.02 mm | 0 | 17,379 |
| 分侧零点对照接触像素数 | 16,935 | 17,063 |
| 实际手指关节位置（mm） | 0.934 | 0.947 |

左右接触掩码作水平镜像对齐后，重叠 IoU 从 0.0 提高到 **0.97686**。给左右 Taxim 实例完全相同的合成高度图，输出 RGB 的最大逐像素差为 **0**，排除了两路光学渲染器本身参数不同的解释。

![左右光学零点诊断对照](/data/nju/Lincao/InsertAnything_tactile/outputs/symmetry_diagnostic_20261012_v2/calibration_comparison.png)

完整结果：[诊断报告](/data/nju/Lincao/InsertAnything_tactile/outputs/symmetry_diagnostic_20261012_v2/report.json)、[同一帧原始及对照数组](/data/nju/Lincao/InsertAnything_tactile/outputs/symmetry_diagnostic_20261012_v2/diagnostic.npz)。

## 为什么 RGB 仍不是逐像素镜像

左右相机朝向相反、上方向一致；当前几何中对应水平镜像，所以原始图像里的边缘出现在相反侧。RGB 颜色还由接触面的局部坡度及传感器自身的彩色照明映射决定，接触轮廓镜像并不要求颜色逐像素镜像。物体偏心、倾斜、L 形或三角形的不同接触面也会产生真实差异。

在对称物体、居中夹持、两侧受压接近且分侧零点正确的情况下，应先比较镜像对齐后的接触轮廓和深度，再比较 RGB。不能通过复制或翻转另一侧图像来生成“对称”的触觉数据。

## 分侧校准已接入运行

当前 `geometry_per_side` 模式从各侧 USD 胶垫前表面计算零点，在生成 Taxim RGB 前应用。对称方形物体实测左右压入 0.238/0.268 mm，接触掩码镜像 IoU 0.97686；移走物体后两侧压入及接触像素均为零。相机裁剪范围保持 24–34 mm。新数据记录原始相机深度和校准 ID，模型与数据执行版本匹配；详见[v2 运行手册](README_insertanything_tactile_pose_v2.md)。

这是刚体胶垫的分侧几何光学零点，尚不是软胶形变、接触力或真实相机的逐像素标定。历史 v1 权重、图像与指标仍对应原来的 28 mm 基准。

现有姿态数据的物体/夹爪实际位姿和 SO(3) 标签不由触觉零点计算；此次发现主要影响触觉外观、几何压入代理和基于该代理的样本有效性筛选。

## 重现短诊断

```bash
cd /data/nju/Lincao/InsertAnything_tactile
./scripts/python.sh scripts/diagnose_tactile_symmetry.py --headless --output outputs/symmetry_diagnostic_new
```

该脚本使用一个方孔环境，执行 8 步，保存原始与分侧零点对照的 RGB、原始深度和报告。输出目录必须是新目录。该历史诊断入口显式使用 `legacy_shared`，再在内存中计算分侧光学对照，不改写资产或默认配置。检验当前真实运行路径请用 `scripts/validate_tactile_calibration.py`，命令见 v2 手册。

早期诊断 `outputs/symmetry_diagnostic_20261012/` 混用了 Fabric 运行时胶垫位姿与相机缓存的 USD 位姿，几何结果无效，已标注 `status=invalid`；以本页的 `symmetry_diagnostic_20261012_v2/` 为准。
