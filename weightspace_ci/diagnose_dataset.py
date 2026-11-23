"""One-shot, bounded Kaggle dataset-upload diagnosis; no GPU or model inference.

Only publish allowlisted status names. Authenticated CLI text and account names
never leave this process. A tiny, synthetic private probe dataset is created
only when the original offline dataset is proven absent.
"""
import json
import os
from pathlib import Path
import re
import tempfile
import time

from weightspace_ci import dispatch


ORIGINAL_RUN = "45051693771"
VALID_STATES = {"ready", "pending", "creating", "processing", "running", "error", "failed"}


def state(ref):
    """Read one Kaggle dataset status without printing the raw API response."""
    raw = dispatch.run_cli("datasets", "status", ref, "--format", "json(status)")
    obj = json.loads(raw)
    value = obj.get("status") if isinstance(obj, dict) else None
    if value not in VALID_STATES:
        raise RuntimeError("Kaggle returned an unrecognized dataset status")
    return value


def make_probe(owner, run_id, root):
    """Prepare public-safe bytes for exactly one private dataset upload."""
    if not re.fullmatch(r"[0-9]+", run_id):
        raise ValueError("Invalid run identity")
    slug = "wsp-upload-probe-" + run_id
    ref = owner + "/" + slug
    (root / "probe.txt").write_text("WSP Kaggle transport probe; no model or personal data.\n",
                                    encoding="utf-8")
    (root / "dataset-metadata.json").write_text(
        json.dumps({"id": ref, "title": slug, "licenses": [{"name": "other"}],
                    "description": "Tiny synthetic private upload diagnostic. No model or personal data."}),
        encoding="utf-8")
    return ref


def diagnose(run_id):
    """Read the previous attempt first; do not duplicate a possibly uploaded dataset."""
    owner = dispatch.kaggle_owner()
    original = owner + "/wsp-offline-" + ORIGINAL_RUN
    try:
        previous = state(original)
    except RuntimeError as error:
        # An unavailable or forbidden lookup must not trigger a write.
        if not str(error).endswith("diagnosis=NOT_FOUND"):
            raise RuntimeError("Original dataset lookup blocked: " + str(error)) from None
        previous = "absent"
    print("ORIGINAL_DATASET=" + previous, flush=True)
    if previous != "absent":
        # A ready original means the 'create failed' report may be client-side;
        # any other state needs investigation before writing anything.
        return previous

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ref = make_probe(owner, run_id, root)
        receipt = dispatch.run_cli(*dispatch.dataset_create_args(root), timeout=180)
        if not dispatch.private_dataset_created(receipt, ref):
            raise RuntimeError("Private probe creation receipt missing; no second upload")
        print("PROBE_PRIVATE_RECEIPT=VERIFIED", flush=True)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            current = state(ref)
            if current == "ready":
                print("PROBE_PRIVATE_DATASET=READY", flush=True)
                return "ready"
            if current not in ("pending", "creating", "processing", "running"):
                raise RuntimeError("Private probe ended without readiness: " + current)
            time.sleep(10)
        raise RuntimeError("Private probe readiness timeout; no retry")


def main():
    # A single, trusted main push only; retries must never upload duplicate data.
    if (os.environ.get("GITHUB_REPOSITORY") != "xxammaxx/mindloop-weightspace-lab"
            or os.environ.get("GITHUB_REF") != "refs/heads/main"
            or os.environ.get("GITHUB_RUN_ATTEMPT") != "1"
            or not os.environ.get("KAGGLE_API_TOKEN")):
        raise RuntimeError("Untrusted context, repeat attempt, or missing Kaggle token")
    diagnose(os.environ["GITHUB_RUN_ID"])


if __name__ == "__main__":
    main()
