# figures/

Place the 8 publication PNGs here (300 DPI), with these exact names (referenced
by the paper and by analysis/make_figures.py):

| File | Content |
|---|---|
| fig1_section_distribution.png | PE section-count distribution by class |
| fig2_entropy_profile.png      | Entropy by section position (+ KS panel) |
| fig3_top_families.png         | Top-20 malware families (family-filtered) |
| fig4_delta_i.png              | Top-20 inter-cell pairs by ΔI |
| fig5_fdr_comparison.png       | Positional vs global-row FDR |
| fig6_prism_heatmaps.png       | FDR / MI heatmaps over the 17×25 lattice |
| fig7a_bodmas_temporal.png     | BODMAS monthly temporal distribution |
| fig7b_bodmas_families.png     | Top-15 BODMAS families |

VERIFY before committing (text baked into images must match paper v1.12):
  - fig1: malware secondary peak at 7 sections (6 is a local minimum).
  - fig2: split line at 2020-07-25 is the 80th percentile (not "median").
