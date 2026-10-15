"""Plot measured RGB and geometric indentation from a saved contact check."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("capture", type=Path)
args = parser.parse_args()
contact = np.load(args.capture / "contact.npz")
removed = np.load(args.capture / "peg_removed.npz")
sides = [side for side in ("left", "right") if f"{side}_rgb" in contact]
fig, axes = plt.subplots(len(sides), 3, figsize=(11, 3.5 * len(sides)), squeeze=False,
                         constrained_layout=True)
vmax = max(float(contact[f"{side}_press_depth_mm"].max()) for side in sides)
for row, side in enumerate(sides):
    axes[row, 0].imshow(contact[f"{side}_rgb"][0])
    depth = axes[row, 1].imshow(contact[f"{side}_press_depth_mm"][0], cmap="magma", vmin=0, vmax=vmax)
    axes[row, 2].imshow(removed[f"{side}_rgb"][0])
    axes[row, 0].set_ylabel(side.capitalize(), fontsize=12)
    for column, title in enumerate(("Contact: Taxim RGB", "Geometric indentation (mm)", "Peg removed: Taxim RGB")):
        axes[row, column].set_title(title, fontsize=11)
        axes[row, column].set_xticks([])
        axes[row, column].set_yticks([])
fig.colorbar(depth, ax=axes[:, 1].tolist(), shrink=0.75, label="mm")
fig.suptitle("GelSight Mini / InsertAnything tactile contact check (environment 0)", fontsize=14)
path = args.capture / "comparison.png"
fig.savefig(path, dpi=150)
print(path.resolve())
