"""Create a 4x2 grid of the top 8 binder visualizations sorted by ipTM."""
import json, re
from pathlib import Path
from PIL import Image

REPO   = Path("/Users/navyanori/hadsbm-hiv")
OUTDIR = REPO / "results/ablations"
CF_DIR = OUTDIR / "colabfold_out_nohup"
VIZ    = OUTDIR / "docking_viz"

# Collect best ipTM per design (same logic as render_all.py)
seen = {}
for f in sorted(CF_DIR.glob("*_scores_rank_*.json")):
    m = re.match(r"(prophet_\d+_rb[\d.]+_wt[\d.]+)_scores_rank_(\d+)_(.*?)\.json", f.name)
    if not m: continue
    design = m.group(1)
    iptm = json.loads(f.read_text()).get("iptm", 0)
    if design not in seen or iptm > seen[design]:
        seen[design] = iptm

ranked = sorted(seen.items(), key=lambda x: -x[1])[:6]
print("Top 8 by ipTM:")
for i, (d, s) in enumerate(ranked, 1):
    print(f"  {i}. {d}  ipTM={s:.3f}")

# Load PNGs
imgs = []
for design, _ in ranked:
    p = VIZ / f"{design}_viz.png"
    if not p.exists():
        print(f"  WARNING: missing {p.name}")
        continue
    imgs.append(Image.open(p).convert("RGB"))

if not imgs:
    raise RuntimeError("No images found")

# Resize each to same width (1200px) preserving aspect ratio
W = 1200
resized = []
for img in imgs:
    h = int(img.height * W / img.width)
    resized.append(img.resize((W, h), Image.LANCZOS))

# 2 columns x 3 rows
COLS, ROWS = 2, 3
PAD = 20
cell_w = W
cell_h = resized[0].height
grid_w = COLS * cell_w + (COLS + 1) * PAD
grid_h = ROWS * cell_h + (ROWS + 1) * PAD

grid = Image.new("RGB", (grid_w, grid_h), (240, 240, 240))
for i, img in enumerate(resized):
    row, col = divmod(i, COLS)
    x = PAD + col * (cell_w + PAD)
    y = PAD + row * (cell_h + PAD)
    grid.paste(img, (x, y))

out = OUTDIR / "docking_viz" / "top8_grid.png"
grid.save(out, dpi=(200, 200))
print(f"\nSaved → {out}")
