"""Frozen public Qwen CPU observation helper for follow-on WSP experiments."""
from contextlib import contextmanager
import os
from pathlib import Path

from weightspace_ci import pilot
from weightspace_ci.prepare_bundle import MODEL_FILES


@contextmanager
def frozen_cpu_observer():
    import numpy as np
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    if transformers.__version__ != "5.12.0" or np.__version__ != "2.2.6":
        raise RuntimeError("Unexpected CPU research dependency versions")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    snapshot = Path(snapshot_download(pilot.MODEL, revision=pilot.REVISION,
                                      allow_patterns=list(MODEL_FILES), token=False))
    if pilot.sha_file(snapshot / pilot.WEIGHTS) != pilot.WEIGHT_SHA:
        raise RuntimeError("Pinned model checksum mismatch")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    torch.set_num_threads(2)
    torch.manual_seed(pilot.SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, trust_remote_code=False, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        snapshot, dtype=torch.float32, trust_remote_code=False,
        local_files_only=True).eval().to("cpu")
    model.requires_grad_(False)
    ids = [tokenizer.encode(x, add_special_tokens=False) for x in pilot.LABELS]
    if any(len(tokens) != 1 for tokens in ids) or len({v[0] for v in ids}) != 4:
        raise RuntimeError("Incompatible output action-token encoding")
    token_ids = [x[0] for x in ids]
    captured = []
    hook = model.get_output_embeddings().register_forward_pre_hook(
        lambda _m, args: captured.append(args[0][0, -1].detach().float().cpu().numpy().copy()))

    def extract(rows, prompt):
        hidden, logits = [], []
        with torch.inference_mode():
            for text, _ in rows:
                captured.clear()
                encoded = tokenizer(prompt + text + "\nAnswer:", return_tensors="pt")
                result = model(**encoded, logits_to_keep=1, use_cache=False)
                if len(captured) != 1:
                    raise RuntimeError("Non-unique output-head observation")
                hidden.append(captured[0])
                logits.append(result.logits[0, -1, token_ids].float().cpu().numpy())
        return np.asarray(hidden), np.asarray(logits)

    try:
        yield extract, torch.__version__, transformers.__version__, np.__version__
    finally:
        hook.remove()
