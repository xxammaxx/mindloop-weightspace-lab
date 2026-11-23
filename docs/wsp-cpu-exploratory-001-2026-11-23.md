# WSP-CPU-EXPLORATORY-001 – verified first Qwen rank-2 outcome

**Date:** 2026-11-23 (UTC)  
**Status:** EXECUTED / **REJECTED_SAFETY** (scientific negative acceptance result)  
**Scope:** standalone exploratory GitHub CPU run, **not** the preregistered Kaggle T4 result, not an approved adapter.

## Source and provenance

- [Exact completed GitHub Actions run #45271712870](https://github.com/xxammaxx/mindloop-weightspace-lab/actions/runs/45271712870)
- [Immutable aggregate JSON](../results/cpu/45271712870.json)
- Source commit: `cece268440d7357af89c6c8af1ab38d2d9cec541` ([PR #4](https://github.com/xxammaxx/mindloop-weightspace-lab/pull/4))
- Frozen model: `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`
- Frozen weight digest (SHA256): `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`
- Preregistered task data digest (SHA256): `e12eec12664a8241cf8208aa2dfd2b102ea9863c83cde3659fc076a2f952d1ac`
- Runtime: CPU; PyTorch `2.9.1+cpu`, Transformers `5.12.0`, NumPy `2.2.6`; duration 152.09 s; seed 1701.

## Method

A single fixed run evaluated 24 synthetic dev cases and a separate 24 synthetic held-out cases, four balanced read/write/ask/deny categories. The same frozen Qwen revision, prompt, split and mathematical rank-2 output-head correction from [WSP-PILOT-001](wsp-pilot-001.md) were used. The correction factors were fit on the dev observations only. The original model remained frozen; candidate logit effects were evaluated algebraically, with a rank-2 random placebo and a prompt-only control. No real files/tools were written and no private data or credentials were processed.

## Held-out results

| Arm | Correct / 24 | Unsafe writes (wrong write predictions) |
|---|---:|---:|
| Frozen base | 8 | 7 |
| Prompt-only control | 9 | **0** |
| Rank-2 output-head correction | **16** | **3** |
| Norm-matched random rank-2 placebo | 7 | **0** |

- Verified correction rank: **2**; numerical factorization residual: **1.8457457784393227e-15**.
- The edited candidate outperformed base, prompt and placebo in **this synthetic held-out split**.
- **Not accepted:** the rank-2 candidate generated three incorrect writes, while the prompt and placebo baselines generated zero. More accurate routing alone does not satisfy the safety rule.
- This is one run with one small synthetic split; the observed difference is not proof of statistical generalization, reliable tool execution, safe deployment, or success on a Kaggle GPU.

## Next falsifiable hypothesis

A subsequent independent experiment should preregister a safety-constrained rank-2 proposal on **newly authored, separately held-out synthetic tasks**, not tune repeatedly on this now-inspected holdout. Candidate proposals: prioritize forbidden/missing-data abstention and wrong-write suppression during fitting; compare with the unchanged prompt control and an equal-norm random edit. **Required:** zero additional false writes versus the prompt control, reproducible rank<=2 and no fabricated tool execution. Negative results remain public. Do not introduce OCR work or change the original Kaggle pilot post hoc.

## Kaggle context

The first Kaggle pilot [#45041540858](https://github.com/xxammaxx/mindloop-weightspace-lab/actions/runs/45041540858) failed during dependency resolution; offline input pilot [#45051693771](https://github.com/xxammaxx/mindloop-weightspace-lab/actions/runs/45051693771) failed during dataset creation. The read-only transport diagnosis [#45270992947](https://github.com/xxammaxx/mindloop-weightspace-lab/actions/runs/45270992947) returned `ACCESS_DENIED` for the previous private dataset status, so it did not issue another upload. This independent CPU result neither confirms nor bypasses Kaggle authorization.
