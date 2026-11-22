# WSP-PILOT-001: direct rank-2 routing probe

Status: preregistered, not yet run on a model. This public companion contains synthetic tasks only. No training, private files, real tool execution, or paid fallback.

## Question and acceptance criteria

Does a closed-form rank-2 correction of the output projection improve constrained next-action selection relative to the same frozen Qwen3.5-0.8B, a prompt-only control and a norm-matched random rank-2 placebo? The primary hypothesis is falsified for this split if the candidate does not outperform both baseline and prompt control; even a gain is not acceptable if it introduces additional incorrect writes. A valid negative result is still an executed experiment.

- Host evidence: algebra reconstructs a known rank-2 edit, placebo norm matches, dev/holdout split disjoint, exact run identity and quota gates tested.
- Target evidence: private Kaggle T4 execution, pinned Apache-2.0 Qwen revision `2fc06364715b967f1860aea9cf38778875588b17`, weight SHA256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`, Transformers 5.12.0, runtime and device recorded.
- Target evidence: 24 synthetic development cases and 24 separately authored held-out cases; four balanced labels A=read, B=write, C=ask, D=deny. Seed 1701. Dataset SHA256 `e12eec12664a8241cf8208aa2dfd2b102ea9863c83cde3659fc076a2f952d1ac`.
- Compare correct/total and wrong write counts across base, prompt, rank-2 and placebo. Require candidate factors rank <=2 and algebra residual <=1e-6; report without claiming generalized tool use or training-free adaptation outside this task.
- Target evidence: validated `metrics.json` attributed to the same GitHub run and source commit. Any missing result is BLOCKED, not a positive or negative model result.

## Method

Collect the last activation at the actual LM head and scores for four single-token response labels. Fit a residual target using a fixed four-logit separation, closed-form ridge in sample space and SVD truncation to rank two. Keep the original model unchanged; selected rows of `W + BA` are evaluated algebraically as the exact corresponding change to the logits. Because Qwen ties the input and output embeddings, do not modify the shared parameter in place. This is a sparse output-head adapter experiment, not a standard PEFT export and not a full published WSP reproduction. Calibration uses only development labels; holdout is evaluated once afterwards without tuning. The base and placebo use the same prompt, the control clarifies rules. No model output is executed.

## Execution envelope

Manual workflow from trusted main only. One in-flight job; 30-minute Kaggle timeout; 35-minute runner timeout; at most three attempts in seven days, no rerun after an unresolved failure. Require authenticated GPU quota >=1.10h, use only the standard NvidiaTeslaT4 accelerator, no paid fallback. A new private kernel per GitHub run avoids stale results. Never pass GitHub or Kaggle credentials to the kernel. Only checked metrics JSON is uploaded as a GitHub artifact. No schedule until this manual end-to-end run is verified.

Sources: [model card and revision](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17), [weight checksum](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/model.safetensors-00001-of-00001.safetensors), [Kaggle CLI kernel documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).
