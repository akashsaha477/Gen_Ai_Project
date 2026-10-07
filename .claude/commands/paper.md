---
description: Update the paper draft from generated artifacts (usage: /paper <section | all>)
---
Section: $ARGUMENTS

1. Read @docs/PAPER_TRACK.md, docs/DECISIONS.md and the current paper/ draft.
2. Write or update the requested section(s) in paper/. Every number must come from paper/generated/ (tables and numbers.tex macros). Never type a number by hand. If a number doesn't exist yet, insert \todo{...}.
3. Match claims to evidence: no claim stronger than the CIs support; unsupported hypotheses reported plainly; limitations specific, not generic.
4. Citations: only entries in paper/refs.bib verified from the DOI or arXiv page. Never invent a reference or a BibTeX field. Mark anything unverified \todo{verify}.
5. Update paper/claims_vs_evidence.md: each claim → table/figure → run IDs.
