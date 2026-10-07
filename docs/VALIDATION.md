# Release validation record

Baseline validation was completed on 2026-09-02. The V18.1 statistical extension and unified regression suite were revalidated on 2026-10-06 against the files retained in this repository.

| Check | Result |
| --- | --- |
| Baseline scientific audit | PASS, 68/68 checks |
| Scientific regression suite, including integer-boundary and statistical-design tests | PASS, 29/29 tests |
| Cluster-bootstrap design-space analysis | PASS, 5,000 replicates over 256 base-geometry clusters |
| Integer-boundary adversarial stress | PASS, 15,360/15,360 classifications stable through ±0.01 module |
| Figure regeneration | PASS, 6 main and 10 supplementary vector PDFs |
| Figure structural and provenance audit | PASS |

The baseline audit verified 3,840 unit cases, 7,680 service cases, 15,360 matched pairwise comparisons, zero system-scale pairwise reversals, 218 equal-integer module pairs, and 52 objective-resolution alternatives. The V18.1 extension independently verifies scale-decomposition identities, bootstrap intervals, integer-boundary stability, and invariance of the 52-member objective-resolution set. Generated PDFs are not committed; regenerate them using the commands in `REPRODUCIBILITY.md`.
