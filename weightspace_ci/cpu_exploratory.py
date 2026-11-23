"""Exploratory WSP rank-2 experiment on a free GitHub CPU runner.

This uses the frozen WSP-PILOT-001 Qwen revision, tasks, labels and rank-2
algebra. Its CPU result is *separate* from the preregistered Kaggle T4 proof.
No tools are executed; only sanitized aggregate counts are saved.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from weightspace_ci import pilot
from weightspace_ci.prepare_bundle import MODEL_FILES


EXPERIMENT = "WSP-CPU-EXPLORATORY-001"
EXPECTED_DATA_SHA = "e12eec12664a8241cf8208aa2dfd2b102ea9863c83cde3659fc076a2f952d1ac"


def score_tables(dev_h, dev_logits, hold_h, hold_logits, prompt_logits):
    """Same four arms and frozen fit/evaluation order as the Kaggle preregistration."""
    import numpy as np
    shape = (len(pilot.DEV), len(pilot.LABELS))
    target_shape = (len(pilot.HOLDOUT), len(pilot.LABELS))
    arrays = [np.asarray(v, dtype=np.float64)
              for v in (dev_h, dev_logits, hold_h, hold_logits, prompt_logits)]
    dev_h, dev_logits, hold_h, hold_logits, prompt_logits = arrays
    if (dev_h.ndim != 2 or hold_h.ndim != 2
            or dev_h.shape[0] != shape[0] or dev_h.shape[1] < 1
            or hold_h.shape != (target_shape[0], dev_h.shape[1])
            or dev_logits.shape != shape
            or hold_logits.shape != target_shape
            or prompt_logits.shape != target_shape
            or not all(np.isfinite(v).all() for v in arrays)):
        raise ValueError("Missing, nonfinite, or mismatched frozen split observations")

    # Identical mathematical candidate specification to pilot.py.
    target = np.repeat((dev_logits.max(axis=1) - 2)[:, None], len(pilot.LABELS), axis=1)
    target[np.arange(len(pilot.DEV)), [y for _, y in pilot.DEV]] += 4
    a, b = pilot.factor_edit(dev_h, target - dev_logits)
    random_edit = pilot.placebo(a, b)
    edited = hold_logits + hold_h @ a.T @ b.T
    placebo_logits = hold_logits + hold_h @ random_edit.T

    expected = [y for _, y in pilot.HOLDOUT]
    m = {name: pilot.metrics(expected, matrix.argmax(axis=1).tolist())
         for name, matrix in (("base", hold_logits), ("prompt", prompt_logits),
                              ("rank2", edited), ("placebo", placebo_logits))}
    m["delta_rank"] = int(np.linalg.matrix_rank(b @ a))
    error = float(np.max(np.abs(hold_h @ a.T @ b.T - hold_h @ (b @ a).T)))
    if m["delta_rank"] not in (1, 2) or not math.isfinite(error) or error > 1e-6:
        raise ValueError("Rank-2 factorization verification failed")
    return m, error


def assert_trusted_context():
    if (os.environ.get("GITHUB_REPOSITORY") != "xxammaxx/mindloop-weightspace-lab"
            or os.environ.get("GITHUB_REF") != "refs/heads/main"
            or os.environ.get("GITHUB_RUN_ATTEMPT") != "1"
            or not re.fullmatch(r"[0-9]+", os.environ.get("GITHUB_RUN_ID", ""))
            or not re.fullmatch(r"[a-f0-9]{40}", os.environ.get("GITHUB_SHA", ""))):
        raise RuntimeError("Only one trusted main GitHub run is allowed")


def run():
    assert_trusted_context()
    start = time.monotonic()
    import numpy as np
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    if transformers.__version__ != "5.12.0" or np.__version__ != "2.2.6":
        raise RuntimeError("Unexpected pinned dependency version")
    data_sha = hashlib.sha256(
        json.dumps([pilot.DEV, pilot.HOLDOUT], ensure_ascii=False).encode()).hexdigest()
    if data_sha != EXPECTED_DATA_SHA:
        raise RuntimeError("Preregistered split identity changed")
    torch.set_num_threads(2)
    torch.manual_seed(pilot.SEED)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    # The public revision is downloaded to this GitHub runner before inference;
    # the model is subsequently loaded with local_files_only=True.
    snapshot = Path(snapshot_download(pilot.MODEL, revision=pilot.REVISION,
                                     allow_patterns=list(MODEL_FILES), token=False))
    if pilot.sha_file(snapshot / pilot.WEIGHTS) != pilot.WEIGHT_SHA:
        raise RuntimeError("Pinned Qwen model checksum differs")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False,
                                               local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        snapshot, dtype=torch.float32, trust_remote_code=False,
        local_files_only=True).eval().to("cpu")
    model.requires_grad_(False)

    label_tokens = [tokenizer.encode(x, add_special_tokens=False) for x in pilot.LABELS]
    if any(len(tokens) != 1 for tokens in label_tokens) or len(
            {tokens[0] for tokens in label_tokens}) != 4:
        raise RuntimeError("Unexpected Qwen label tokenization")
    ids = [tokens[0] for tokens in label_tokens]
    captured = []
    hook = model.get_output_embeddings().register_forward_pre_hook(
        lambda _module, args: captured.append(args[0][0, -1].detach().float().cpu().numpy().copy()))

    def extract(rows, prompt):
        hidden, logits = [], []
        with torch.inference_mode():
            for task, _ in rows:
                captured.clear()
                encoded = tokenizer(prompt + task + "\nAnswer:", return_tensors="pt")
                output = model(**encoded, logits_to_keep=1, use_cache=False)
                if len(captured) != 1:
                    raise RuntimeError("Output-head activation capture failed")
                hidden.append(captured[0])
                logits.append(output.logits[0, -1, ids].float().cpu().numpy())
        return np.array(hidden), np.array(logits)

    try:
        dev_h, dev_logits = extract(pilot.DEV, pilot.BASE)
        # Holdout is evaluated only after fitting is frozen.
        hold_h, hold_logits = extract(pilot.HOLDOUT, pilot.BASE)
        _, prompt_logits = extract(pilot.HOLDOUT, pilot.CONTROL)
        measured, residual = score_tables(dev_h, dev_logits, hold_h, hold_logits, prompt_logits)
    finally:
        hook.remove()
    report = {
        "schema": 1, "experiment": EXPERIMENT, "context": "github-public-cpu-exploratory",
        "run_id": os.environ["GITHUB_RUN_ID"], "source_sha": os.environ["GITHUB_SHA"],
        "model": pilot.MODEL, "revision": pilot.REVISION,
        "weights_sha256": pilot.WEIGHT_SHA, "data_sha256": data_sha,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "numpy": np.__version__, "device": "cpu", "seed": pilot.SEED,
        "dev_count": len(pilot.DEV), "holdout_count": len(pilot.HOLDOUT),
        "metrics": measured, "factorization_error": residual,
        "duration_seconds": round(time.monotonic() - start, 2),
    }
    output = Path("cpu-results")
    output.mkdir(exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, sort_keys=True,
                                                   indent=2, allow_nan=False))
    print("WSP CPU exploratory completed; aggregate metrics saved. Not Kaggle T4 evidence.",
          flush=True)


if __name__ == "__main__":
    run()
