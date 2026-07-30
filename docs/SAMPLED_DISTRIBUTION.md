# Answer distribution — sampled LLaVA-OV runs

**These are SAMPLED runs** (`temperature=0.7, top_p=0.9, seed=0`), not the gated greedy results. They cannot pass the divergence gate — two sampled runs differ by chance — so they are not comparable to the numbers in `master-results.md` and must not be merged into it.

Counts are over the 4,018 scoreable questions (NA excluded). *unparsed* = generations with no recoverable A–D letter.

| Method | A | B | C | D | unparsed | most-picked |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| *ground truth* | *25.4%* | *25.6%* | *25.5%* | *23.6%* | *—* | *—* |
| **baseline** | 29.3% | 25.3% | 25.5% | 19.9% | 6 | A 29% |
| **DyCoke** | 29.0% | 25.7% | 25.4% | 19.8% | 6 | A 29% |
| **HoliTom** | 27.2% | 25.0% | 25.7% | 22.1% | 6 | A 27% |
| **MDP3** | 28.9% | 24.7% | 25.9% | 20.4% | 3 | A 29% |
| **AIM** | 29.6% | 24.7% | 25.9% | 19.7% | 6 | A 30% |
| **VideoITG** | 28.9% | 25.1% | 25.1% | 20.8% | 2 | A 29% |
| **STTM** | 29.4% | 24.4% | 25.8% | 20.5% | 6 | A 29% |

⚠️ = one letter takes >40% of answers, i.e. the model is leaning on a default rather than discriminating. Ground truth is near-uniform, so a healthy method should be too.

*Not yet run: FlashVID.*
