"""Render all registered tactile tasks serially, then assemble actual scene images."""
import argparse
import ast
from datetime import datetime
import html
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TITLES = {'CircleHole-I': '圆孔 I', 'SquareHole-II': '方孔 II',
          'LHole-III': 'L 孔 III', 'TriangleHole-IV': '三角孔 IV',
          'HexagonHole-III': '六角孔 III', 'HexagonHole-IV': '六角孔 IV'}
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
VIEWS = [('detail.png', '夹爪、插销与孔位'), ('hole.png', '孔位近景'),
         ('left_tactile.png', '左触觉 RGB'), ('right_tactile.png', '右触觉 RGB')]


def registered_tasks():
    tree = ast.parse((ROOT / 'source/insertanything_tactile/__init__.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TASKS' for t in node.targets):
            return [ast.literal_eval(key) for key in node.value.keys]
    raise RuntimeError('Could not read the current TASKS registry')


def read_reports(output, suffixes):
    records = []
    for suffix in suffixes:
        folder = output / suffix
        report = json.loads((folder / 'report.json').read_text())
        assert report['status'] == 'complete', suffix
        assert report['task'] == f'InsertAnythingTactile-{suffix}-Direct-v0', suffix
        assert report['same_physical_instant_for_all_views_and_tactile'], suffix
        for name in ['scene.png', *[v[0] for v in VIEWS]]:
            pixels = np.asarray(Image.open(folder / name))
            assert pixels.ndim == 3 and pixels.shape[2] == 3 and pixels.std() > 0, (suffix, name)
        with np.load(folder / 'snapshot.npz') as data:
            assert all(np.isfinite(data[key]).all() for key in data.files), suffix
            for side in ('left', 'right'):
                assert np.array_equal(np.asarray(Image.open(folder / f'{side}_tactile.png')), data[f'{side}_rgb'])
                press = data[f'{side}_press_depth_mm']
                assert int(np.count_nonzero(press > 0.02)) == report['contact'][side]['contact_pixels_above_0_02_mm']
        records.append({'suffix': suffix, 'name': TITLES.get(suffix, suffix), 'folder': folder, 'report': report})
    return records


def add_images(fig, rows, columns, output, title):
    grid = fig.add_gridspec(len(rows), len(columns), left=0.015, right=0.985,
                           top=0.89, bottom=0.045, hspace=0.30, wspace=0.055,
                           width_ratios=[1.3, 1, 1, 1] if len(columns) == 4 else None)
    fig.suptitle(title, fontproperties=FONT, fontsize=21, y=0.985)
    fig.text(0.5, 0.94, '同一物理时刻的真实仿真渲染 · 初始化夹持 · GelSight Mini / Taxim',
             ha='center', fontproperties=FONT, fontsize=12, color='#475569')
    for r, record in enumerate(rows):
        for c, (file, label) in enumerate(columns):
            ax = fig.add_subplot(grid[r, c])
            ax.imshow(Image.open(record['folder'] / file))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            caption = f"{record['name']} · {label}" if c < 2 else label
            ax.set_title(caption, fontproperties=FONT, fontsize=13, pad=7)
            if 'tactile' in file:
                side = 'left' if file.startswith('left') else 'right'
                info = record['report']['contact'][side]
                state = '达到接触阈值' if info['geometric_contact_valid'] else '弱接触 / 未达阈值'
                ax.text(0.5, -0.055, f"{state} · 几何压入最大 {info['max_press_depth_mm']:.3f} mm",
                        ha='center', va='top', transform=ax.transAxes, fontproperties=FONT,
                        fontsize=10, color='#334155' if info['geometric_contact_valid'] else '#b45309')
    fig.savefig(output, dpi=170, facecolor='white')
    plt.close(fig)


def assemble(output, records):
    for start in range(0, len(records), 3):
        rows = records[start:start+3]
        page = start // 3 + 1
        fig = plt.figure(figsize=(15.8, 4.15 * len(rows) + 0.8))
        add_images(fig, rows, VIEWS, output / f'scenes_and_tactile_{page}.png',
                   f'InsertAnything + tactile 场景与双侧触觉 · {page}/{(len(records)+2)//3}')
    fig, axes = plt.subplots((len(records)+1)//2, 2, figsize=(14, 4.9*((len(records)+1)//2)))
    for ax, record in zip(np.asarray(axes).flat, records):
        ax.imshow(Image.open(record['folder'] / 'scene.png'))
        ax.axis('off')
        ax.set_title(f"{record['name']} · {record['suffix']}", fontproperties=FONT, fontsize=15)
    for ax in list(np.asarray(axes).flat)[len(records):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(output / 'all_scenes_overview.png', dpi=150, facecolor='white')
    plt.close(fig)
    for record in records:
        fig = plt.figure(figsize=(15.8, 5.2))
        add_images(fig, [record], VIEWS, record['folder'] / 'scene_and_tactile.png',
                   f"{record['name']} · {record['suffix']}")
    cards = []
    for record in records:
        report = record['report']
        folder = html.escape(record['suffix'])
        images = []
        for file, title in [('scene.png', '机械臂全景'), *VIEWS]:
            images.append(f'<figure><a href="{folder}/{file}"><img loading="lazy" src="{folder}/{file}" alt="{html.escape(title)}"></a><figcaption>{html.escape(title)}</figcaption></figure>')
        contacts = '；'.join(f"{'左' if side == 'left' else '右'}：{'达到接触阈值' if report['contact'][side]['geometric_contact_valid'] else '弱接触 / 未达阈值'}，最大几何压入 {report['contact'][side]['max_press_depth_mm']:.3f} mm" for side in ('left', 'right'))
        cards.append(f'<section data-task="{folder}" id="{folder}"><h2>{html.escape(record["name"])}</h2><code>{html.escape(report["task"])}</code><p>seed {report["seed"]}，初始化后 {report["control_steps_after_reset"]} 步；插销标称尺寸 {report["held_asset"]["nominal_size_mm"]:.3f} mm，孔标称尺寸 {report["fixed_asset"]["nominal_size_mm"]:.3f} mm。</p><div class="images">{"".join(images)}</div><p>{contacts}</p><p><a href="{folder}/snapshot.npz">原始双侧 RGB / 深度与位姿 NPZ</a> · <a href="{folder}/report.json">采样与配置报告</a></p></section>')
    options = ''.join(f'<option value="{html.escape(r["suffix"])}">{html.escape(r["name"])}</option>' for r in records)
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>InsertAnything + tactile 场景图集</title><style>body{font:16px/1.65 sans-serif;margin:30px auto;max-width:1700px;padding:0 20px;background:#f8fafc;color:#172033}h1{font-size:30px}section{background:white;padding:22px;border-radius:14px;margin:25px 0;border:1px solid #dde3eb}.images{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}figure{margin:0}img{width:100%;aspect-ratio:4/3;object-fit:contain;background:#eef2f6}a{color:#1261a0}select{padding:8px;min-width:180px}code{overflow-wrap:anywhere}@media(max-width:900px){.images{grid-template-columns:repeat(2,1fr)}}</style><h1>InsertAnything + tactile：全部场景</h1><p>真实仿真截图，采用初始化后的夹持状态。每个场景的不同相机视角与双侧触觉来自同一物理时刻；改变外部相机期间没有推进物理仿真。触觉为 GelSight Mini / Taxim / PhysX 刚体胶垫输出。</p><p>接触阈值：至少 64 个像素的几何压入超过 0.02 mm；该深度不是标定后的压力或接触力。六角 III / IV 使用不同插销尺寸与任务配置，外观接近。</p><label>场景筛选：<select id="filter"><option value="all">显示全部</option>'+options+'</select></label>'+''.join(cards)+'<script>document.querySelector("#filter").addEventListener("change",e=>document.querySelectorAll("section").forEach(s=>s.hidden=e.target.value!=="all"&&s.dataset.task!==e.target.value))</script></html>'
    (output / 'index.html').write_text(document)
    summary = {'status': 'complete', 'registered_scene_count': len(records),
               'tasks': [r['report'] for r in records],
               'validation': 'All five PNGs are nonempty; stereo PNGs equal raw NPZ; states finite; contact counts match',
               'scope': 'Initialized physical grasp snapshots, not success-rate evaluation or insertion rollout'}
    (output / 'gallery.json').write_text(json.dumps(summary, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Reuse completed, matching captures in an existing output')
    parser.add_argument('--assemble_only', action='store_true', help='Regenerate figures from existing captures, without Isaac Sim')
    args = parser.parse_args()
    if args.steps < 1 or (args.assemble_only and args.output is None):
        parser.error('Positive --steps required; --assemble_only also requires --output')
    output = (args.output or ROOT / 'outputs' / datetime.now().strftime('scene_gallery_%Y%m%d_%H%M%S')).resolve()
    if not args.assemble_only:
        output.mkdir(parents=True, exist_ok=args.resume)
    suffixes = registered_tasks()
    for index, suffix in enumerate(suffixes):
        folder = output / suffix
        if args.assemble_only:
            continue
        if folder.exists():
            report = json.loads((folder / 'report.json').read_text())
            assert report['status'] == 'complete' and report['seed'] == args.seed and report['control_steps_after_reset'] == args.steps
            print(f'GALLERY_REUSE {index+1}/{len(suffixes)} {suffix}', flush=True)
            continue
        command = [sys.executable, str(ROOT / 'scripts/render_scene.py'), '--task',
                   f'InsertAnythingTactile-{suffix}-Direct-v0', '--steps', str(args.steps),
                   '--seed', str(args.seed), '--output', str(folder)]
        if args.headless:
            command.append('--headless')
        print(f'GALLERY_RENDER {index+1}/{len(suffixes)} {suffix}', flush=True)
        with (output / f'{suffix}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Capture failed: {suffix}; see {output / (suffix + ".log")}')
        print(f'GALLERY_CAPTURED {index+1}/{len(suffixes)} {suffix}', flush=True)
    records = read_reports(output, suffixes)
    assemble(output, records)
    print(f'GALLERY_COMPLETE {len(records)} scenes {output}', flush=True)


if __name__ == '__main__':
    main()
