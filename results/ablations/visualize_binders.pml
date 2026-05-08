# PROPHET HIV-1 Protease Binder Visualization
# Top binder: prophet_0003 (ipTM=0.180), peptide SLEMYFGCHE
# Chain A = peptide, Chain B = protease

reinitialize

load /Users/navyanori/hadsbm-hiv/results/ablations/colabfold_out_nohup/prophet_0003_rb0.22_wt0.24_unrelaxed_rank_001_alphafold2_multimer_v3_model_5_seed_000.pdb, cmplx

# ── Selections ────────────────────────────────────────────────────────────────
select ligand, cmplx and chain A
select target, cmplx and chain B
select res_sites, target and resi 10+11+13+14+16+20+23+24+30+32+33+34+35+36+43+45+46+47+48+50+53+54+58+60+62+63+64+71+73+76+77+82+83+84+85+88+89+90+93
select iface, target within 5.0 of ligand
select res_iface, res_sites and iface

# ── Style (from inspiration) ──────────────────────────────────────────────────
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

# ── Representations ───────────────────────────────────────────────────────────
hide everything, cmplx

# Target: ribbon inside transparent surface
show cartoon, target
show surface, target
color lightblue, target
set transparency, 0.5, target

# Resistance sites: colored on surface/cartoon (salmon)
color salmon, res_sites

# Interface residues at resistance sites: highlighted red
color red, res_iface

# Ligand: sticks + cartoon on top, paleyellow
show cartoon, ligand
show sticks,  ligand
color paleyellow, ligand
set stick_radius, 0.2

# ── Overview ──────────────────────────────────────────────────────────────────
orient cmplx
zoom cmplx, 4
turn x, -20
turn y, 30

ray 2400, 2400
png /Users/navyanori/hadsbm-hiv/results/ablations/prophet_0003_overview.png, dpi=300

# ── Interface close-up ────────────────────────────────────────────────────────
zoom iface, 10
orient iface
turn y, 20
turn x, -10

ray 2400, 2400
png /Users/navyanori/hadsbm-hiv/results/ablations/prophet_0003_interface.png, dpi=300
