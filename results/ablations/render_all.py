"""Render labeled PyMOL visualizations for all ColabFold designs with PDBs."""
import json, re, math, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO   = Path("/Users/navyanori/hadsbm-hiv")
OUTDIR = REPO / "results/ablations"
CF_DIR = OUTDIR / "colabfold_out_nohup"
PYMOL  = "/Users/navyanori/miniconda3/bin/pymol"

RES_SITES = {10,11,13,14,16,20,23,24,30,32,33,34,35,36,43,45,46,47,48,50,
             53,54,58,60,62,63,64,71,73,76,77,82,83,84,85,88,89,90,93}

THREE = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
         'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
         'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}


def get_designs():
    seen = {}
    for f in sorted(CF_DIR.glob("*_scores_rank_*.json")):
        m = re.match(r"(prophet_\d+_rb[\d.]+_wt[\d.]+)_scores_rank_(\d+)_(.*?)\.json", f.name)
        if not m: continue
        design, rank, mstr = m.group(1), int(m.group(2)), m.group(3)
        pdb = CF_DIR / f"{design}_unrelaxed_rank_{rank:03d}_{mstr}.pdb"
        if not pdb.exists(): continue
        iptm = json.loads(f.read_text()).get("iptm", 0)
        if design not in seen or iptm > seen[design]["iptm"]:
            seen[design] = {"iptm": iptm, "pdb": pdb}
    return sorted(seen.items(), key=lambda x: -x[1]["iptm"])


def get_peptide_and_interface(pdb_path):
    coords, seq_a, seq_b = {}, {}, {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"): continue
            ch = line[21]; ri = int(line[22:26]); r3 = line[17:20].strip()
            ax,ay,az = float(line[30:38]),float(line[38:46]),float(line[46:54])
            coords[(ch,ri,line[12:16].strip())] = (ax,ay,az)
            if ch=='A': seq_a[ri] = r3
            if ch=='B': seq_b[ri] = r3
    pep_atoms = {k:v for k,v in coords.items() if k[0]=='A'}
    iface = set()
    for (cb,rb,ab),c in coords.items():
        if cb!='B': continue
        for pa in pep_atoms.values():
            if math.sqrt(sum((a-b)**2 for a,b in zip(c,pa))) < 5.0:
                iface.add(rb); break
    peptide = "".join(THREE.get(v,"?") for v in seq_a.values())
    res_iface = sorted(RES_SITES & iface)
    return peptide, res_iface, seq_b


def render_pymol(pdb_abs, out_png):
    res_str = "+".join(str(r) for r in RES_SITES)
    pml = f"""
reinitialize
load {pdb_abs}, cmplx
select ligand, cmplx and chain A
select target, cmplx and chain B
select res_sites, target and resi {res_str}
select iface, target within 5.0 of ligand
select res_iface, res_sites and iface
bg_color white
set ray_opaque_background, on
set specular, 0
set ambient, 0.4
set cartoon_highlight_color, grey
set ray_trace_depth_factor, 1
set ray_trace_disco_factor, 1
set ray_trace_mode, 1
set antialias, 2
set ray_shadow, off
set cartoon_loop_radius, 0.25
set cartoon_fancy_helices, 1
hide everything, cmplx
show cartoon, target
show surface, target
color lightblue, target
set transparency, 0.5, target
color salmon, res_sites
color red, res_iface
show cartoon, ligand
show sticks, ligand
color paleyellow, ligand
set stick_radius, 0.2
orient cmplx
zoom cmplx, 4
turn x, -20
turn y, 30
ray 2400, 2400
png {out_png}, dpi=300
"""
    tmp = OUTDIR / "_tmp_render.pml"
    tmp.write_text(pml)
    subprocess.run([PYMOL, "-c", str(tmp)], capture_output=True)
    tmp.unlink()


def annotate(png_path, design, peptide, iptm, res_iface, seq_b, rank):
    img = Image.open(png_path).convert("RGBA")
    W, H = img.size
    ov = Image.new("RGBA", (W,H), (0,0,0,0))
    draw = ImageDraw.Draw(ov)

    try:
        fl = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 52)
        fm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        fs = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except:
        fl = fm = fs = ImageFont.load_default()

    # Top banner
    draw.rectangle([(0,0),(W,155)], fill=(255,255,255,220))
    draw.text((36,14),  f"#{rank}  {design}",      font=fl, fill=(20,20,80,255))
    draw.text((36,78),  f"Peptide: {peptide}",     font=fm, fill=(40,40,40,255))
    draw.text((680,78), f"ipTM: {iptm:.3f}",       font=fm, fill=(40,40,40,255))

    # Bottom bar — resistance sites
    draw.rectangle([(0,H-72),(W,H)], fill=(255,255,255,220))
    if res_iface:
        site_strs = [f"{THREE.get(seq_b.get(r,''),'?')}{r}" for r in res_iface]
        line = "Resistance sites at interface:  " + "  ·  ".join(site_strs)
        col = (180,20,20,255)
    else:
        line = "No resistance sites at interface"
        col = (100,100,100,255)
    draw.text((36, H-56), line, font=fs, fill=col)

    # Legend
    lx, ly = W-310, 165
    draw.rectangle([(lx,ly),(W-20,ly+130)], fill=(255,255,255,210))
    boxes = [("lightblue","HIV-1 protease",(173,216,230)),
             ("paleyellow","Binder peptide",(250,235,150)),
             ("salmon","Resistance sites",(250,128,114)),
             ("red","Res. at interface",(220,20,20))]
    for i,(_, label, rgb) in enumerate(boxes):
        y = ly+8+i*28
        draw.rectangle([(lx+8,y+2),(lx+30,y+20)], fill=(*rgb,255))
        draw.text((lx+38,y-2), label, font=fs, fill=(20,20,20,255))

    out = Image.alpha_composite(img, ov).convert("RGB")
    out.save(png_path)


# ── Main ─────────────────────────────────────────────────────────────────────
designs = get_designs()
print(f"Rendering {len(designs)} designs...\n")

for rank, (design, info) in enumerate(designs, 1):
    out_png = OUTDIR / f"{design}_viz.png"
    peptide, res_iface, seq_b = get_peptide_and_interface(info["pdb"])
    print(f"  [{rank:02d}/{len(designs)}] {design}  ipTM={info['iptm']:.3f}  pep={peptide}  res_iface={res_iface}")
    render_pymol(str(info["pdb"]), str(out_png))
    annotate(out_png, design, peptide, info["iptm"], res_iface, seq_b, rank)

print(f"\nDone — {len(designs)} images in results/ablations/")
