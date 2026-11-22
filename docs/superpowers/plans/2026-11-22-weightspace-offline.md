# Weightspace offline inputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove an internet-disabled Kaggle notebook can load the exact WSP-PILOT-001 model and dependency wheels before a single bounded T4 attempt.

**Architecture:** GitHub builds and hashes a flat private version-1 Kaggle dataset from public upstream files. Both a CPU qualification kernel and the existing GPU experiment mount that exact dataset and verify its manifest, installing from local wheels with no index. GPU launch requires validated CPU evidence and the existing weekly quota guard.

**Tech Stack:** Python 3.12/3.13, pip, GitHub Actions, Kaggle CLI 2.2.5, Transformers 5.12.0.

**Spec:** `docs/wsp-pilot-001.md` and `docs/wsp-pilot-001-run-45041540858.md`; user approved offline packaging on 2026-11-22. Scientific evaluation remains identical.

## Global Constraints
- Public model and dependencies only; no credentials in dataset/kernel/artifact. Private dataset and CPU/GPU kernels. No paid resources. Same revision/checksum, fixtures and four arms. CPU qualification is not GPU scientific evidence.
- One new GPU attempt at most after the exact failed run 45041540858 is resolved by CPU qualification. Existing three-attempt/7-day and authenticated quota >=1.10h guards remain; any other failed run blocks. No automatic rerun.
- Dataset creation confirms private success and ready version 1; notebook checks exact manifest digest and every file before offline install. No remote Python code. Kaggle notebook internet disabled.

## Review Focus
- corrupt or extra input file: fail closed; missing wheel: no network fallback; stale/mismatched CPU evidence: no GPU; CLI exit zero with error: no GPU; earlier unrecognized failures: count and block.

### Task 1: Bundle identity and offline boot
- [ ] RED: fixture manifest validates, mutation/extra/missing files rejected, install argv is offline-only.
- [ ] Implement in `weightspace_ci/offline.py`, package builder in `weightspace_ci/prepare_bundle.py`. Combine helper and pilot into standalone Kaggle script. Pin manifest digest to kernel, retain upstream licenses. Support CPython 3.12 and 3.13 Linux wheels from native runner pip.
- [ ] GREEN: `python -m unittest discover -s weightspace_ci/tests -v`.

### Task 2: CPU qualification and GPU gate
- [ ] RED: exact CPU run/source/manifest identity and one real forward, dataset private create confirmation, known prior failure allowance only after qualification, other failure/duplicate blocked.
- [ ] Implement `weightspace_ci/dispatch.py`, offline mode in `pilot.py`: run CPU smoke with one dev case, output qualification JSON; GPU run same local bundle, existing metrics validator. Verify version 1, ready state and CPU result before GPU. Keep all raw authenticated command output private.
- [ ] GREEN: full test suite; compile and parse workflows.

### Task 3: Public source publication and live qualification
- [ ] Update `.github/workflows/kaggle-pilot.yml`: build wheelhouse and model with secrets absent, private Kaggle upload then CPU first, quota-checked GPU only if verified. CPU bootstrap remains credential-free.
- [ ] Validate readback of exact main, trigger workflow once, record GitHub run and target evidence. Stop on failure; no automatic repeat. Preserve failed first attempt.
