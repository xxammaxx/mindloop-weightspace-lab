"""Trusted GitHub runner orchestrator. Only a single private, bounded T4 job."""
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.request

from weightspace_ci.offline import MANIFEST_NAME, file_sha, validate_bundle


def quota_ok(quota):
    if not isinstance(quota, list):
        return False
    rows = [r for r in quota if isinstance(r, dict) and r.get('resource') == 'GPU']
    if len(rows) != 1:
        return False
    value = rows[0].get('remaining')
    return isinstance(value, str) and bool(re.fullmatch(r'\d+\.\d{2}h', value)) and float(value[:-1]) >= 1.10


def parse_status(text, kernel):
    match = re.search(r'^' + re.escape(kernel) + r' has status "(?:KernelWorkerStatus\.)?([A-Za-z]+)"$', text, re.M)
    return match.group(1).upper() if match else None


def push_version(text, kernel):
    match = re.search(r'^Kernel version ([1-9]\d*) successfully pushed\.  Please check progress at https://(?:www\.)?kaggle\.com/code/' + re.escape(kernel) + r'(?:/|\?[^\s]*)?$', text, re.M)
    return int(match.group(1)) if match else None


def validate_metrics(data, run_id, source_sha, bundle_sha=None):
    allowed = {'schema','experiment','run_id','source_sha','model','revision','weights_sha256','data_sha256','device','torch','transformers','seed','dev_count','holdout_count','metrics','factorization_error','duration_seconds','peak_vram_bytes'}
    if bundle_sha is not None:
        allowed.add('bundle_sha256')
        if not isinstance(data,dict) or data.get('bundle_sha256') != bundle_sha:
            raise ValueError('Unexpected offline bundle identity')
    if not isinstance(data, dict) or set(data) != allowed or data.get('run_id') != run_id or data.get('source_sha') != source_sha:
        raise ValueError('Missing or stale metrics identity')
    if (data['schema'],data['experiment'],data['model'],data['revision'],data['weights_sha256'],data['seed'],data['dev_count'],data['holdout_count']) != (1,'WSP-PILOT-001','Qwen/Qwen3.5-0.8B','2fc06364715b967f1860aea9cf38778875588b17','04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696',1701,24,24):
        raise ValueError('Unexpected model or fixtures')
    if data['device'] not in ('Tesla T4','NVIDIA Tesla T4','NVIDIA T4') or data['transformers'] != '5.12.0' or not re.fullmatch(r'[\w.+-]{1,50}',data['torch']) or data['data_sha256'] != 'e12eec12664a8241cf8208aa2dfd2b102ea9863c83cde3659fc076a2f952d1ac':
        raise ValueError('Unexpected runtime')
    m = data['metrics']
    if set(m) != {'base','prompt','rank2','placebo','delta_rank'} or type(m['delta_rank']) is not int or not 0 < m['delta_rank'] <= 2:
        raise ValueError('Unexpected rank/metric arms')
    for key in ('base','prompt','rank2','placebo'):
        row = m[key]
        if set(row) != {'correct','total','unsafe_write'} or any(type(v) is not int for v in row.values()) or row['total'] != 24 or not 0 <= row['correct'] <= 24 or not 0 <= row['unsafe_write'] <= 18:
            raise ValueError('Invalid metrics')
    for name, maximum in [('factorization_error',1e-6),('duration_seconds',1800),('peak_vram_bytes',16*1024**3)]:
        v = data[name]
        if type(v) not in (int,float) or not math.isfinite(v) or not 0 <= v <= maximum:
            raise ValueError('Invalid numerical evidence')
    return data


def classify_cli_failure(raw):
    """Return only allowlisted failure categories, never authenticated CLI output."""
    signatures = {
        'AUTH_REJECTED': r'\b401\b|unauthori[sz]ed|invalid (?:api )?token|authentication required',
        'ACCESS_DENIED': r'\b403\b|forbidden|permission denied',
        'BAD_REQUEST': r'\b400\b|bad request|validation error|invalid (?:request|argument)',
        'NOT_FOUND': r'\b404\b|not found',
        'ALREADY_EXISTS': r'\b409\b|already exists|duplicate dataset|conflict',
        'INPUT_TOO_LARGE': r'\b413\b|payload too large|request entity too large|file too large',
        'QUOTA_LIMIT': r'\b429\b|rate limit|quota exceeded|storage limit|too many requests',
        'SERVER_ERROR': r'\b50[0-9]\b|internal server error|service unavailable|bad gateway',
        'NETWORK_DNS': r'NameResolutionError|Temporary failure in name resolution|Name or service not known',
        'NETWORK': r'ConnectionError|Connection refused|Network is unreachable|Max retries exceeded|SSLError',
        'TIMEOUT': r'TimeoutExpired|ReadTimeout|ConnectTimeout|timed out',
        'DATASET_METADATA': r'dataset-metadata\.json|invalid metadata|invalid license|dataset title',
        'UPLOAD_ERROR': r'upload failed|failed to upload|upload error',
    }
    matches = sorted(name for name, pattern in signatures.items() if re.search(pattern, raw, re.I))
    return matches or ['UNKNOWN']


def dataset_create_args(root):
    """Private by default. Do not suppress the success receipt needed for verification."""
    return ('datasets', 'create', '-p', str(root), '--keep-tabular')


def run_cli(*args, timeout=120):
    r = subprocess.run(['kaggle', *args], capture_output=True, text=True, timeout=timeout,
                       env={k:v for k,v in os.environ.items() if k!='GH_TOKEN'})
    if r.returncode:
        # Only disclose allowlisted categories; Kaggle may return account data or tokens.
        categories = classify_cli_failure((r.stderr or '') + '\n' + (r.stdout or ''))
        raise RuntimeError('Kaggle CLI command failed: ' + '/'.join(args[:2]) +
                           '; diagnosis=' + ','.join(categories))
    return r.stdout


def summarize_kernel_log(raw):
    """Return only enumerated categories and bounded source line numbers, never text from logs."""
    patterns = {
        'dependency_install': r'No matching distribution found|Could not find a version that satisfies|ResolutionImpossible|pip.*returned non-zero|subprocess\.CalledProcessError',
        'dependency_import': r'ImportError:|cannot import name|undefined symbol|numpy\.dtype size changed',
        'missing_module': r'ModuleNotFoundError:',
        'model_access': r'RepositoryNotFoundError|GatedRepoError|RevisionNotFoundError|EntryNotFoundError|(?:401|403) Client Error',
        'network': r'ConnectionError|Connection refused|Temporary failure in name resolution|ReadTimeout|Network is unreachable|Max retries exceeded',
        'network_dns': r'Temporary failure in name resolution|Name or service not known|NameResolutionError|nodename nor servname provided',
        'network_refused': r'Connection refused|Connection reset|Network is unreachable',
        'network_timeout': r'ReadTimeout|ConnectTimeout|timed out',
        'network_tls': r'SSLError|CERTIFICATE_VERIFY_FAILED|certificate verify failed',
        'network_proxy': r'ProxyError|Tunnel connection failed',
        'package_not_found': r'No matching distribution found|Could not find a version that satisfies',
        'pip_index': r'pypi\.org|files\.pythonhosted\.org|/simple/transformers/',
        'out_of_memory': r'CUDA out of memory|OutOfMemoryError|Cannot allocate memory',
        'gpu_unavailable': r'Required free T4 GPU unavailable|CUDA driver version is insufficient|CUDA error: no kernel image',
        'model_checksum': r'Model checksum differs|FileNotFoundError:.*safetensors',
        'model_runtime': r'AttributeError:|TypeError:|ValueError:|NotImplementedError:|CUDA error:',
        'timeout': r'TimeoutExpired|exceeded.*time limit',
        'tokenization': r'Label tokenization incompatible',
        'hook': r'Output head hook did not capture unique activation',
        'nonfinite': r'Nonfinite edit',
        'process_killed': r'Killed|exit code 137|SIGKILL',
    }
    categories = sorted(k for k, pattern in patterns.items() if re.search(pattern, raw, re.I))
    lines = {int(n) for n in re.findall(r'File "[^"\n]*\bpilot\.py", line ([1-9]\d{0,3})\b', raw)}
    return {'categories': categories or ['unknown'], 'pilot_lines': sorted(lines)}


def kaggle_owner():
    users = re.findall(r'^- username: ([a-zA-Z0-9_-]+)$', run_cli('config','view'), re.M)
    if len(users)!=1 or users[0].lower()=='none':
        raise RuntimeError('Could not identify Kaggle account')
    return users[0]


def diagnose_kernel(kernel):
    try:
        raw = run_cli('kernels','logs',kernel)
    except (RuntimeError, subprocess.TimeoutExpired):
        print('Kaggle diagnostic: log retrieval unavailable', flush=True)
        return
    print('Kaggle diagnostic:', json.dumps(summarize_kernel_log(raw)), flush=True)


def budget_for_runs(runs, total, run_id, qualified):
    if len(runs) != total or not 1 <= len(runs) <= 3 or len({r['id'] for r in runs}) != len(runs):
        return False
    if not any(str(r['id']) == run_id and r.get('run_attempt') == 1 for r in runs):
        return False
    for r in runs:
        if r.get('run_attempt') != 1:
            return False
        if str(r['id']) == run_id:
            continue
        if r.get('status') != 'completed':
            return False
        if r.get('conclusion') == 'success':
            continue
        if not (qualified and r['id'] == 45041540858 and r.get('conclusion') == 'failure'):
            return False
    return True


def weekly_budget(run_id, qualified=False):
    # Count all dispatch attempts conservatively, even unsuccessful ones, in a rolling seven days.
    since = (dt.datetime.now(dt.timezone.utc)-dt.timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%SZ')
    url = 'https://api.github.com/repos/xxammaxx/mindloop-weightspace-lab/actions/workflows/kaggle-pilot.yml/runs?per_page=100&created=%3E%3D'+since
    request = urllib.request.Request(url, headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json','User-Agent':'mindloop-weightspace-ci'})
    with urllib.request.urlopen(request,timeout=20) as response:
        data=json.load(response)
    return budget_for_runs(data['workflow_runs'],data['total_count'],run_id,qualified)


def private_dataset_created(text, reference):
    return bool(re.search(r'^Your private Dataset is being created\. Please check progress at https://(?:www\.)?kaggle\.com/datasets/'+re.escape(reference)+r'/?$',text,re.M))


def validate_qualification(data, run_id, source_sha, bundle_sha):
    expected = {'schema':1,'experiment':'WSP-PILOT-001','run_id':run_id,'source_sha':source_sha,
                'bundle_sha256':bundle_sha,'model':'Qwen/Qwen3.5-0.8B','revision':'2fc06364715b967f1860aea9cf38778875588b17',
                'weights_sha256':'04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696',
                'transformers':'5.12.0','device':'cpu','forward_pass':True,'labels':4}
    if not isinstance(data,dict) or set(data) != {*expected,'torch','hidden_size'} or any(data[k] != v for k,v in expected.items()):
        raise ValueError('Missing or stale offline CPU qualification')
    if data['forward_pass'] is not True or type(data['hidden_size']) is not int or not 0 < data['hidden_size'] <= 16384 or not isinstance(data['torch'],str) or not re.fullmatch(r'[\w.+-]{1,50}',data['torch']):
        raise ValueError('Invalid CPU qualification')
    return data


def create_dataset(root, owner, run_id, digest):
    validate_bundle(root,digest)
    slug = 'wsp-offline-'+run_id
    ref = owner+'/'+slug
    metadata = {'id':ref,'title':slug,'licenses':[{'name':'other'}],
                'description':'Private offline reproducibility input. Qwen model: Apache-2.0, included LICENSE; Python wheels retain individual upstream license files. Exact upstream revision and all file hashes are in repository and manifest.'}
    (root/'dataset-metadata.json').write_text(json.dumps(metadata))
    try:
        result = run_cli(*dataset_create_args(root), timeout=900)
    finally:
        (root/'dataset-metadata.json').unlink()
    if not private_dataset_created(result,ref):
        raise RuntimeError('Private dataset creation unconfirmed; no retry')
    print('Private offline dataset creation confirmed; awaiting ready status.',flush=True)
    deadline = time.monotonic()+300
    while time.monotonic() < deadline:
        status = json.loads(run_cli('datasets','status',ref,'--format','json(status)'))['status']
        if status == 'ready':
            version = json.loads(run_cli('datasets','status',ref,'--format','json(current_version_number)'))['current_version_number']
            if type(version) is not int or version != 1:
                raise RuntimeError('Unexpected dataset version')
            return ref+'/1'
        if status not in ('pending','creating','processing','running'):
            raise RuntimeError('Dataset not ready; no retry')
        time.sleep(10)
    raise RuntimeError('Dataset readiness timeout; no retry')


def run_kernel(owner, run_id, sha, dataset, digest, mode):
    gpu = mode == 'gpu'
    slug = 'mindloop-wsp-pilot-'+run_id if gpu else 'mindloop-wsp-offline-cpu-'+run_id
    kernel = owner+'/'+slug
    timeout = 1800 if gpu else 600
    filename = 'metrics.json' if gpu else 'qualification.json'
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)/'kernel'; folder.mkdir()
        helper=Path(__file__).with_name('offline.py').read_text()
        source=Path(__file__).with_name('pilot.py').read_text().replace('__RUN_ID__',run_id).replace('__SOURCE_SHA__',sha).replace('__BUNDLE_SHA__',digest).replace('__EXEC_MODE__',mode)
        (folder/'pilot.py').write_text(helper+'\n\n'+source)
        metadata={'id':kernel,'title':slug,'code_file':'pilot.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':gpu,'enable_internet':False,'dataset_sources':[dataset],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
        args=['kernels','push','-p',str(folder),'--timeout',str(timeout)]
        if gpu:
            metadata['machine_shape']='NvidiaTeslaT4'
            args+=['--accelerator','NvidiaTeslaT4']
        (folder/'kernel-metadata.json').write_text(json.dumps(metadata))
        if push_version(run_cli(*args),kernel) != 1:
            raise RuntimeError('Kaggle push did not confirm new unique kernel version 1; no retry')
        kernel += '/1'
        print(f'Private {mode} kernel dispatched; monitoring.',flush=True)
        deadline=time.monotonic()+timeout
        previous=None
        while time.monotonic()<deadline:
            status=parse_status(run_cli('kernels','status',kernel),kernel)
            if status!=previous:
                print('Kernel status:',status or 'UNKNOWN',flush=True); previous=status
            if status=='COMPLETE':
                break
            if status not in ('QUEUED','RUNNING'):
                if status == 'ERROR': diagnose_kernel(kernel)
                raise RuntimeError('Kaggle kernel not successful; no retry')
            time.sleep(20)
        else:
            raise RuntimeError('Monitoring deadline exceeded; do not relaunch')
        output=Path(tmp)/'output'; output.mkdir()
        run_cli('kernels','output',kernel,'-p',str(output),'--file-pattern','^'+re.escape(filename)+'$')
        path=output/filename
        if not path.is_file() or path.stat().st_size>16384:
            raise RuntimeError('Missing or oversized result; no publish')
        return json.loads(path.read_text())


def main():
    run_id, sha = os.environ['GITHUB_RUN_ID'], os.environ['GITHUB_SHA']
    if not re.fullmatch(r'\d+',run_id) or not re.fullmatch(r'[a-f0-9]{40}',sha) or os.environ.get('GITHUB_RUN_ATTEMPT')!='1':
        raise RuntimeError('Unexpected run identity or duplicate attempt')
    if not os.environ.get('KAGGLE_API_TOKEN') or not weekly_budget(run_id,qualified=True):
        raise RuntimeError('Missing token or preparation budget blocked')
    root=Path(os.environ['WSP_BUNDLE_DIR'])
    digest=file_sha(root/MANIFEST_NAME)
    owner=kaggle_owner()
    dataset=create_dataset(root,owner,run_id,digest)
    qualified=validate_qualification(run_kernel(owner,run_id,sha,dataset,digest,'cpu'),run_id,sha,digest)
    dest=Path('pilot-results'); dest.mkdir(exist_ok=True)
    (dest/'qualification.json').write_text(json.dumps(qualified,indent=2,allow_nan=False))
    print('Verified target CPU offline forward pass; checking GPU budget.',flush=True)
    if not weekly_budget(run_id,qualified=True) or not quota_ok(json.loads(run_cli('quota','--format','json'))):
        raise RuntimeError('No verified free GPU quota or weekly budget; no GPU dispatch')
    report=validate_metrics(run_kernel(owner,run_id,sha,dataset,digest,'gpu'),run_id,sha,digest)
    (dest/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('Validated WSP-PILOT-001 metrics:',json.dumps(report['metrics']),flush=True)


if __name__=='__main__':
    main()
