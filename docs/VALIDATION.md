# Release validation record

Validation was run in a locked Python environment on 2026-09-02 against the files retained in this repository.

| Check | Result |
| --- | --- |
| Baseline scientific audit | PASS, 68/68 checks |
| Scientific regression suite, including V18 integer-boundary tests | PASS, 23/23 tests |
| Figure regeneration | PASS, 6 main and 9 supplementary vector PDFs |
| Figure structural and provenance audit | PASS |

The baseline audit verified 3,840 unit cases, 7,680 service cases, 15,360 matched pairwise comparisons, zero system-scale pairwise reversals, 218 baseline equal-integer module pairs, and 52 objective-resolution alternatives. The generated PDFs are not committed; regenerate them using the commands in `REPRODUCIBILITY.md`.
