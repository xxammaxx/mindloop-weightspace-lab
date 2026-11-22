"""Shared, standalone offline bundle validation; no network access here."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

MANIFEST_NAME = 'wsp-input-manifest.json'
REQUIREMENTS = ('transformers==5.12.0', 'numpy==2.2.6')


def file_sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_bundle(root, expected_sha):
    root = Path(root)
    try:
        raw = (root/MANIFEST_NAME).read_bytes()
        if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError('Offline manifest checksum differs')
        data = json.loads(raw)
        if set(data) != {'schema','files'} or data['schema'] != 1 or not isinstance(data['files'],dict) or not data['files']:
            raise ValueError('Invalid offline manifest')
        files = data['files']
        if set(p.name for p in root.iterdir()) != {*files, MANIFEST_NAME}:
            raise ValueError('Missing or unexpected offline bundle files')
        for name, digest in files.items():
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.+-]*',name) or name == MANIFEST_NAME or not re.fullmatch(r'[a-f0-9]{64}',digest):
                raise ValueError('Invalid offline manifest entry')
            if not (root/name).is_file() or file_sha(root/name) != digest:
                raise ValueError('Offline input checksum differs')
        return files
    except (OSError, TypeError, KeyError) as exc:
        raise ValueError('Invalid or missing offline input') from exc


def install_argv(root, target):
    return [sys.executable,'-m','pip','--isolated','install','--disable-pip-version-check',
            '--no-index','--only-binary=:all:','--ignore-installed','--find-links',str(root),
            '--target',str(target),*REQUIREMENTS]


def bootstrap_offline(expected_sha, input_dir='/kaggle/input'):
    manifests = list(Path(input_dir).rglob(MANIFEST_NAME))
    if len(manifests) != 1:
        raise ValueError('Missing or ambiguous offline manifest')
    root = manifests[0].parent
    validate_bundle(root, expected_sha)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    target = tempfile.mkdtemp(prefix='wsp-deps-')
    subprocess.run(install_argv(root,target),check=True,timeout=300)
    sys.path.insert(0,target)
    print('Verified offline bundle; local dependency installation finished.',flush=True)
    return root
