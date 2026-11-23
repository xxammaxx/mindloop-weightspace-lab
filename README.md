# MindLoop Weightspace Lab

Public, reproducible research into **direct low-rank changes to small language models' output behavior**, including task-specific tool-use/routing adapters. The aim is to test weight-space programming on synthetic tasks before any deployment. This is **not** an OCR research repository.

## First real model result — 2026-11-23

**[WSP-CPU-EXPLORATORY-001: completed, REJECTED_SAFETY](docs/wsp-cpu-exploratory-001-2026-11-23.md)**

| Frozen Qwen3.5-0.8B / 24 held-out synthetic tasks | Correct | Unsafe writes |
|---|---:|---:|
| Base model | 8 | 7 |
| Prompt-only control | 9 | 0 |
| Calculated rank-2 output-head edit | **16** | **3** |
| Random rank-2 placebo | 7 | 0 |

The direct mathematical edit improved accuracy **on this one split**, but **failed the no-new-unsafe-write acceptance criterion**. It is not a qualified adapter, not proof of generalization, and not a Kaggle GPU result. No real tools were executed.

**Public evidence:** [exact GitHub Actions run](https://github.com/xxammaxx/mindloop-weightspace-lab/actions/runs/45271712870), [machine-readable metrics](results/cpu/45271712870.json), [methods and limitations](docs/wsp-cpu-exploratory-001-2026-11-23.md).

## Second real model result — 2026-11-23

**[WSP-CPU-SAFETY-002: completed, LAB_CANDIDATE](docs/wsp-cpu-safety-002-2026-11-23.md)**

A **new**, independently authored 24-case held-out routing cohort (not the first study's holdout) gave: base **10/24** (8 wrong-write predictions); prompt **6/24** (0); ordinary rank-2 **18/24** (0); safety-biased rank-2 **12/24** (0); placebo **6/24** (0).

The safety-biased edit passed its predeclared local lab gate, **but ordinary rank-2 also had zero wrong-write predictions and higher accuracy on the same new cohort**. Therefore this does **not** prove the safety penalty improves the model. No real tools were executed.

**Public evidence:** [exact GitHub Actions run](https://github.com/xxammaxx/mindloop-weightspace-lab/actions/runs/45273653240), [machine-readable metrics](results/cpu/safety-002-45273653240.json), [methods, scientific limitations](docs/wsp-cpu-safety-002-2026-11-23.md), [predeclared issue](https://github.com/xxammaxx/mindloop-weightspace-lab/issues/6).

## Reproducibility and next steps

- [Original preregistered Kaggle T4 study](docs/wsp-pilot-001.md): same frozen Qwen revision, 24 development cases, 24 separate holdout cases, and four arms. **Kaggle remains infrastructure-blocked**, not falsified.
- [Current research and infrastructure issue](https://github.com/xxammaxx/mindloop-weightspace-lab/issues/1).
- Next scientific priority: a **safety-constrained** rank-2 hypothesis with a new independent holdout, zero extra unsafe writes, and falsifiable gates. Never optimize to the now revealed holdout.
- Public GitHub CI runs algebra and provenance tests; a completed trusted CPU study publishes only verified aggregate metrics under `results/cpu/`. Failed and negative results stay documented.
- No private documents, secrets, full model dumps, paid inference, or unapproved software/tool actions are part of this project.
