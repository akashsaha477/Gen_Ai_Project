# Course report

The project's course report and its results.

| File | What it holds |
|---|---|
| [report.pdf](report.pdf) | The course report |
| [results.md](results.md) | Every results table of the report, readable on GitHub |
| [report.tex](report.tex), [refs.bib](refs.bib) | The report's LaTeX source and its references |
| [generated/](generated/) | The tables, number macros and results text the report inputs |

Nothing in `generated/` or `results.md` is typed by hand. Both are written by `python -m ecgaug.paper` from the result
files under [`../results/`](../results/), which hold the summary of every run; the figures come from
[`../figures/`](../figures/).

Rebuild from the repository root:

```bash
python -m ecgaug.paper              # regenerates report/generated/ and report/results.md
cd report && tectonic report.tex    # compiles report.pdf
```
