"""GitHub-only: package public pinned model plus already downloaded wheels."""
import json
from pathlib import Path
import shutil
import sys

from weightspace_ci.offline import MANIFEST_NAME, file_sha, validate_bundle
from weightspace_ci.pilot import MODEL, REVISION, WEIGHTS, WEIGHT_SHA

MODEL_FILES = ('LICENSE','README.md','chat_template.jinja','config.json','merges.txt',WEIGHTS,
               'model.safetensors.index.json','preprocessor_config.json','tokenizer.json',
               'tokenizer_config.json','video_preprocessor_config.json','vocab.json')


def prepare(root):
    from huggingface_hub import snapshot_download
    root = Path(root)
    if not root.is_dir() or not any(root.glob('transformers-5.12.0-*.whl')) or any(not p.is_file() or p.suffix != '.whl' for p in root.iterdir()):
        raise RuntimeError('Expected wheel-only staging directory')
    snapshot = Path(snapshot_download(MODEL,revision=REVISION,allow_patterns=list(MODEL_FILES),token=False))
    for name in MODEL_FILES:
        shutil.copyfile(snapshot/name,root/name)
    if file_sha(root/WEIGHTS) != WEIGHT_SHA:
        raise RuntimeError('Model checksum differs')
    files = {p.name:file_sha(p) for p in sorted(root.iterdir())}
    (root/MANIFEST_NAME).write_text(json.dumps({'schema':1,'files':files},sort_keys=True),encoding='utf-8')
    digest = file_sha(root/MANIFEST_NAME)
    validate_bundle(root,digest)
    print(f'Offline input prepared: {len(files)} files, {sum(p.stat().st_size for p in root.iterdir())} bytes; manifest sha256 {digest}',flush=True)


if __name__ == '__main__':
    prepare(sys.argv[1])
