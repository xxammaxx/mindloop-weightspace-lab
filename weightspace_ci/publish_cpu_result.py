"""Validate and publish one aggregate-only WSP CPU research artifact.

Invoked only by a trusted workflow_run completion; no Kaggle credentials,
model prompts, raw model outputs or user files are ingested.
"""
import io
import json
import math
import os
from pathlib import Path
import re
import urllib.request
import zipfile

from weightspace_ci import cpu_exploratory as cpu, pilot


REPO = "xxammaxx/mindloop-weightspace-lab"
MAX_ZIP = 65536
MAX_JSON = 16384


def validate_report(data, run_id, source_sha):
    expected_keys = {"schema", "experiment", "context", "run_id", "source_sha",
                     "model", "revision", "weights_sha256", "data_sha256",
                     "transformers", "torch", "numpy", "device", "seed",
                     "dev_count", "holdout_count", "metrics",
                     "factorization_error", "duration_seconds"}
    if (not isinstance(data, dict) or set(data) != expected_keys
            or data["schema"] != 1 or data["experiment"] != cpu.EXPERIMENT
            or data["context"] != "github-public-cpu-exploratory"
            or data["run_id"] != run_id or data["source_sha"] != source_sha
            or data["model"] != pilot.MODEL or data["revision"] != pilot.REVISION
            or data["weights_sha256"] != pilot.WEIGHT_SHA
            or data["data_sha256"] != cpu.EXPECTED_DATA_SHA
            or data["transformers"] != "5.12.0" or data["numpy"] != "2.2.6"
            or data["device"] != "cpu" or data["seed"] != pilot.SEED
            or data["dev_count"] != 24 or data["holdout_count"] != 24):
        raise ValueError("Untrusted CPU research evidence identity")
    if not isinstance(data["torch"], str) or not re.fullmatch(r"[A-Za-z0-9.+_-]{1,60}", data["torch"]):
        raise ValueError("Invalid CPU runtime")
    metrics = data["metrics"]
    if (not isinstance(metrics, dict)
            or set(metrics) != {"base", "prompt", "rank2", "placebo", "delta_rank"}
            or type(metrics["delta_rank"]) is not int
            or metrics["delta_rank"] not in (1, 2)):
        raise ValueError("Invalid or absent CPU research arms")
    for name in ("base", "prompt", "rank2", "placebo"):
        v = metrics[name]
        if (not isinstance(v, dict)
                or set(v) != {"correct", "total", "unsafe_write"}
                or any(type(z) is not int for z in v.values())
                or v["total"] != 24 or not 0 <= v["correct"] <= 24
                or not 0 <= v["unsafe_write"] <= 18):
            raise ValueError("Invalid CPU aggregate counts")
    for field, maximum in (("factorization_error", 1e-6),
                           ("duration_seconds", 2700)):
        v = data[field]
        if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= maximum:
            raise ValueError("Invalid CPU experiment numerical evidence")
    return data


def extract_report(blob):
    if not isinstance(blob, bytes) or len(blob) > MAX_ZIP:
        raise ValueError("Missing or oversized evidence archive")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        entries = [x for x in z.infolist() if not x.is_dir()]
        if len(entries) != 1 or entries[0].filename != "metrics.json" or entries[0].file_size > MAX_JSON:
            raise ValueError("Unexpected evidence archive content")
        raw = z.read(entries[0])
        if len(raw) > MAX_JSON:
            raise ValueError("Oversized evidence metrics")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Malformed evidence metrics")
        return data


def get_json(url, token):
    req = urllib.request.Request(
        url, headers={"Authorization": "Bearer " + token,
                      "Accept": "application/vnd.github+json",
                      "User-Agent": "wsp-cpu-evidence-publisher"})
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.load(res)


class UnauthenticatedArtifactRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward a GitHub bearer token to a signed artifact storage URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith("https://"):
            raise ValueError("Unsafe artifact redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
        return redirected


def get_zip(url, token):
    req = urllib.request.Request(
        url, headers={"Authorization": "Bearer " + token,
                      "Accept": "application/vnd.github+json",
                      "User-Agent": "wsp-cpu-evidence-publisher"})
    opener = urllib.request.build_opener(UnauthenticatedArtifactRedirect())
    with opener.open(req, timeout=30) as res:
        blob = res.read(MAX_ZIP + 1)
    if len(blob) > MAX_ZIP:
        raise ValueError("Oversized evidence download")
    return blob


def main():
    run_id = os.environ.get("SOURCE_RUN_ID", "")
    source_sha = os.environ.get("SOURCE_SHA", "")
    token = os.environ.get("GH_TOKEN", "")
    if (os.environ.get("GITHUB_REPOSITORY") != REPO
            or not re.fullmatch(r"[0-9]+", run_id)
            or not re.fullmatch(r"[a-f0-9]{40}", source_sha)
            or not token):
        raise RuntimeError("Untrusted source identity")
    path = Path("results/cpu") / (run_id + ".json")
    if path.exists():
        raise RuntimeError("CPU evidence already published; never overwrite")
    url = "https://api.github.com/repos/" + REPO + "/actions/runs/" + run_id + "/artifacts?per_page=100"
    artifacts = get_json(url, token)
    expected = "wsp-cpu-exploratory-" + run_id
    rows = [row for row in artifacts.get("artifacts", [])
            if isinstance(row, dict) and row.get("name") == expected
            and row.get("expired") is False]
    if len(rows) != 1 or type(rows[0].get("id")) is not int:
        raise RuntimeError("Missing or ambiguous exact CPU evidence artifact")
    zip_url = "https://api.github.com/repos/" + REPO + "/actions/artifacts/" + str(rows[0]["id"]) + "/zip"
    report = validate_report(extract_report(get_zip(zip_url, token)), run_id, source_sha)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("PUBLIC_CPU_EVIDENCE=VERIFIED; WSP-CPU-EXPLORATORY-001; not Kaggle GPU.", flush=True)


if __name__ == "__main__":
    main()
