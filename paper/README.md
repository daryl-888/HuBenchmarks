# paper/

Technical report on the MotionBench token-reduction benchmark.

`paper.tex` is the source of truth. Every number in it is recomputed from
`results-cache/*/results.jsonl` — nothing is transcribed from prose.

## Rebuild

```bash
python3 paper/build_data.py       # recompute all numbers -> paper/data.json
python3 paper/make_figures2.py    # regenerate the four figures from data.json
cd paper && pdflatex paper.tex && pdflatex paper.tex   # twice, for refs/ToC
```

Both Python scripts resolve paths relative to the repo, so they run anywhere
it is checked out. Override with `$HUVLLM_CACHE` and `$HUVLLM_DATA` if the
results cache lives elsewhere.

## Conventions

- Rounding is **half away from zero** throughout. Two cells sit on an exact
  tie (109/400 and 115/400 and 89/400 = x.x5); Python's default banker's
  rounding disagrees with the paper on these, so do not "fix" them to match
  `build_data.py` output without changing the stated convention first.
- Deltas are computed against the **unrounded** baseline (52.6630%, not
  52.66%). Using the rounded value shifts STTM from −0.95 to −0.94.
- Significance is McNemar on paired predictions, χ² ≥ 3.84. The ±1.54-point
  binomial floor at n=4,018 governs what may be called a difference at all.

## Note on paper.pdf

`paper.pdf` is committed for convenience but **no LaTeX toolchain is
available in this environment**, so it can lag `paper.tex`. If the two
disagree, `paper.tex` is correct. Rebuild before circulating the PDF.
