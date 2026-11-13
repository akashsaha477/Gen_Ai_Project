# Course report

The project's course report, in two versions, and its results.

| File | What it holds |
|---|---|
| [report.pdf](report.pdf) | The course report, full version (29 pages) |
| [report_short.pdf](report_short.pdf) | The same report in 20 pages: without the related work, how each part was chosen, the discussion and the reproducibility section, and with fewer figures |
| [results.md](results.md) | Every results table of the report, readable on GitHub |
| [report.tex](report.tex), [report_short.tex](report_short.tex), [refs.bib](refs.bib) | The LaTeX sources of both versions and their references |
| [generated/](generated/) | The tables, number macros and results text the two versions input |

Nothing in `generated/` or `results.md` is typed by hand. Both are written by `python -m ecgaug.paper` from the result
files under [`../results/`](../results/), which hold the summary of every run; the figures come from
[`../figures/`](../figures/).

Rebuild from the repository root:

```bash
python -m ecgaug.paper              # regenerates report/generated/ and report/results.md
cd report && tectonic report.tex    # compiles report.pdf
tectonic report_short.tex           # compiles report_short.pdf
```
