# Manuscript source

Everything needed to rebuild the CICADAS manuscript and its supplementary material, so that the
paper is reproducible from this repository alongside the code that produced its results.

## Build

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
pdflatex supplementary_material.tex && bibtex supplementary_material \
  && pdflatex supplementary_material.tex && pdflatex supplementary_material.tex
```

The supplement uses `\externaldocument{main}` (package `xr`) for cross-document references, so
`main.aux` must exist before the supplement is built. Build the main document first.

Compiled PDFs (`main.pdf`, `supplementary_material.pdf`) are included so the paper can be read
without a LaTeX installation.

## Where the numbers come from

| Manuscript element | Produced by |
|---|---|
| Fig. 1 exemplar trajectories | `../matlab/a0_GenerateTrialData.m` |
| Fig. 4 swimmer plots | `../CICADA_FIGURES/a3a_*`, `a3b_*` |
| Fig. 5 PK/PD recovery | `../matlab/a1_EstimatePKPD.m` |
| Fig. 6 g-formula recovery | `../CICADA_FIGURES/a3c_Fig_Swimmers_Obs_g_formula.m` |
| Fig. 7 policy sweep and heatmaps | `../matlab/a4_*` |
| Table 1 multi-seed replication | `../multiseed/` |
| Table 2 estimator benchmarking | `../benchmarks/` |
| Robustness analyses | `../sensitivity/` |

## Note

Figure files are the rendered PDFs used at submission. Regenerating them from the scripts above will
produce cohort-level differences, since each run draws a new simulated cohort; see `../multiseed/`
for the sampling distributions that the paper's claims rest on.
