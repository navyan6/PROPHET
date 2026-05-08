"""
Renders PyMOL PNGs for top 3 binders, then annotates with matplotlib labels.
"""
import subprocess, math, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO = Path("/Users/navyanori/hadsbm-hiv")
OUTDIR = REPO / "results/ablations"
PYMOL = "/Users/navyanori/miniconda3/bin/pymol"

RES_SITES = {10,11,13,14,16,20,23,24,30,32,33,34,35,36,43,45,46,47,48,50,
             53,54,58,60,62,63,64,71,73,76,77,82,83,84,85,88,89,90,93}

BINDERS = [
    {"design": "prophet_0003", "iptm": 0.180, "peptide": "SLEMYFGCHE",
     "pdb": "colabfold_out_nohup/prophet_0003_rb0.22_wt0.24_unrelaxed_rank_001_alphafold2_multimer_v3_model_5_seed_000.pdb"},
    {"design": "prophet_0005", "iptm": 0.120, "peptide": "FWDFDLQGRG",
     "pdb": "colabfold_out_nohup/prophet_0005_rb0.22_wt0.24_unrelaxed_rank_001_alphafold2_multimer_v3_model_4_seed_000.pdb"},
    {"design": "prophet_0006", "iptm": 0.110, "peptide": "ERTGIVYDWP",
     "pdb": "colabfold_out_nohup/prophet_0006_rb0.22_wt0.24_unrelaxed_rank_001_alphafold2_multimer_v3_model_4_seed_000.pdb"},
]

THREE_TO_ONE = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
                'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
                'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

RES_NAMES = {10:'I10',11:'V11',13:'I13',14:'L14',16:'K16',20:'K20',23:'L23',
             24:'L24',30:'D30',32:'V32',33:'L33',34:'E34',35:'E35',36:'M36',
             43:'I43',45:'F53',46:'M46',47:'I47',48:'G48',50:'I50',53:'F53',
             54:'I54',58:'Q58',60:'D60',62:'I62',63:'L63',64:'I64',71:'A71',
             73:'G73',76:'V76',77:'V77',82:'V82',83:'N83',84:'I84',85:'I85',
             88:'N88',89:'L89',90:'L90',93:'I93'}


def get_interface(pdb_path):
    coords = {}
    seq_b = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"): continue
            chain = line[21]
            resi  = int(line[22:26].strip())
            res3  = line[17:20].strip()
            aname = line[12:16].strip()
            x,y,z = float(line[30:38]),float(line[38:46]),float(line[46:54])
            coords[(chain,resi,aname)] = (x,y,z)
            if chain == 'B': seq_b[resi] = res3

    pep_atoms = {k:v for k,v in coords.items() if k[0]=='A'}
    iface = set()
    for (cb,rb,ab),coord in coords.items():
        if cb != 'B': continue
        for pa in pep_atoms.values():
            if math.sqrt(sum((a-b)**2 for a,b in zip(coord,pa))) < 5.0:
                iface.add(rb)
                break
    return sorted(RES_SITES & iface), seq_b


def make_pml(binder, out_png):
    pdb_abs = str(OUTDIR / binder["pdb"])
    res_str = "+".join(str(r) for r in RES_SITES)
    return f"""
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


def annotate(png_path, binder, res_iface, seq_b):
    img = Image.open(png_path).convert("RGBA")
    W, H = img.size

    overlay = Image.new("RGBA", (W, H), (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    try:
        font_lg = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 52)
        font_md = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 38)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    except Exception:
        font_lg = font_md = font_sm = ImageFont.load_default()

    # Semi-transparent top banner
    draw.rectangle([(0,0),(W,160)], fill=(255,255,255,210))

    # Title
    draw.text((40, 18),  f"{binder['design']}",             font=font_lg, fill=(20,20,80,255))
    draw.text((40, 82),  f"Peptide: {binder['peptide']}",   font=font_md, fill=(40,40,40,255))
    draw.text((700, 82), f"ipTM: {binder['iptm']:.3f}",     font=font_md, fill=(40,40,40,255))

    # Resistance sites at interface
    if res_iface:
        site_strs = []
        for r in res_iface:
            aa1 = THREE_TO_ONE.get(seq_b.get(r,''),'?')
            site_strs.append(f"{aa1}{r}")
        sites_line = "Resistance sites at interface: " + ", ".join(site_strs)
        draw.rectangle([(0, H-80),(W,H)], fill=(255,255,255,210))
        draw.text((40, H-62), sites_line, font=font_sm, fill=(180,20,20,255))
    else:
        draw.rectangle([(0, H-80),(W,H)], fill=(255,255,255,210))
        draw.text((40, H-62), "No resistance sites at interface", font=font_sm, fill=(100,100,100,255))

    # Legend
    draw.rectangle([(W-340, 170),(W-20, 310)], fill=(255,255,255,200))
    draw.rectangle([(W-320, 190),(W-290, 210)], fill=(173,216,230,255))
    draw.text((W-280, 186), "HIV-1 protease",  font=font_sm, fill=(20,20,20,255))
    draw.rectangle([(W-320, 220),(W-290, 240)], fill=(250,235,150,255))
    draw.text((W-280, 216), "Binder peptide",  font=font_sm, fill=(20,20,20,255))
    draw.rectangle([(W-320, 250),(W-290, 270)], fill=(250,128,114,255))
    draw.text((W-280, 246), "Resistance sites", font=font_sm, fill=(20,20,20,255))
    draw.rectangle([(W-320, 280),(W-290, 300)], fill=(220,20,20,255))
    draw.text((W-280, 276), "Res. at interface", font=font_sm, fill=(20,20,20,255))

    out = Image.alpha_composite(img, overlay).convert("RGB")
    out.save(png_path)
    print(f"  Annotated → {png_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "pillow", "-q"])
    from PIL import Image, ImageDraw, ImageFont

for b in BINDERS:
    print(f"\nRendering {b['design']} (peptide={b['peptide']}, ipTM={b['iptm']})...")
    out_png = str(OUTDIR / f"{b['design']}_viz.png")
    pml = make_pml(b, out_png)
    pml_path = OUTDIR / f"_tmp_{b['design']}.pml"
    pml_path.write_text(pml)
    result = subprocess.run([PYMOL, "-c", str(pml_path)], capture_output=True, text=True)
    if "wrote" not in result.stderr and "wrote" not in result.stdout:
        print(f"  PyMOL stderr: {result.stderr[-300:]}")
    pml_path.unlink()
    res_iface, seq_b = get_interface(str(OUTDIR / b["pdb"]))
    print(f"  Resistance sites at interface: {res_iface}")
    annotate(out_png, b, res_iface, seq_b)

print("\nDone. Output:")
for b in BINDERS:
    print(f"  results/ablations/{b['design']}_viz.png")
