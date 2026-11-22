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


def validate_metrics(data, run_id, source_sha):
    allowed = {'schema','experiment','run_id','source_sha','model','revision','weights_sha256','data_sha256','device','torch','transformers','seed','dev_count','holdout_count','metrics','factorization_error','duration_seconds','peak_vram_bytes'}
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


def run_cli(*args):
    r = subprocess.run(['kaggle', *args], capture_output=True, text=True, timeout=120,
                       env={k:v for k,v in os.environ.items() if k!='GH_TOKEN'})
    if r.returncode:
        # Never publish output from an authenticated command.
        raise RuntimeError('Kaggle CLI command failed: ' + '/'.join(args[:2]))
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


def weekly_budget(run_id):
    # Count all dispatch attempts conservatively, even unsuccessful ones, in a rolling seven days.
    since = (dt.datetime.now(dt.timezone.utc)-dt.timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%SZ')
    url = 'https://api.github.com/repos/xxammaxx/mindloop-weightspace-lab/actions/workflows/kaggle-pilot.yml/runs?per_page=100&created=%3E%3D'+since
    request = urllib.request.Request(url, headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json','User-Agent':'mindloop-weightspace-ci'})
    with urllib.request.urlopen(request,timeout=20) as response:
        data=json.load(response)
    runs=data['workflow_runs']
    return (len(runs)==data['total_count'] and len(runs)<=3 and any(str(r['id'])==run_id for r in runs)
            and all(str(r['id'])==run_id or (r.get('status')=='completed' and r.get('conclusion')=='success') for r in runs))


def main():
    run_id, sha = os.environ['GITHUB_RUN_ID'], os.environ['GITHUB_SHA']
    if not re.fullmatch(r'\d+',run_id) or not re.fullmatch(r'[a-f0-9]{40}',sha) or os.environ.get('GITHUB_RUN_ATTEMPT')!='1':
        raise RuntimeError('Unexpected run identity or duplicate attempt')
    if not os.environ.get('KAGGLE_API_TOKEN') or not weekly_budget(run_id):
        raise RuntimeError('Missing token or weekly budget blocked')
    if not quota_ok(json.loads(run_cli('quota','--format','json'))):
        raise RuntimeError('No verified free GPU quota >= 1.10 hours; no dispatch')
    slug='mindloop-wsp-pilot-'+run_id
    kernel=kaggle_owner()+'/'+slug
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)/'kernel'; folder.mkdir()
        source=Path(__file__).with_name('pilot.py').read_text().replace('__RUN_ID__',run_id).replace('__SOURCE_SHA__',sha)
        (folder/'pilot.py').write_text(source)
        metadata={'id':kernel,'title':slug,'code_file':'pilot.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
        (folder/'kernel-metadata.json').write_text(json.dumps(metadata))
        pushed=run_cli('kernels','push','-p',str(folder),'--accelerator','NvidiaTeslaT4','--timeout','1800')
        version=push_version(pushed,kernel)
        if version!=1:
            raise RuntimeError('Kaggle push did not confirm new unique kernel version 1; no retry')
        kernel += '/1'
        print('Kaggle private pilot dispatch returned; monitoring.',flush=True)
        deadline=time.monotonic()+30*60
        previous=None
        while time.monotonic()<deadline:
            status=parse_status(run_cli('kernels','status',kernel),kernel)
            if status!=previous:
                print('Kernel status:',status or 'UNKNOWN',flush=True); previous=status
            if status=='COMPLETE':
                break
            if status not in ('QUEUED','RUNNING'):
                if status == 'ERROR':
                    diagnose_kernel(kernel)
                raise RuntimeError('Kaggle pilot not successful; no retry')
            time.sleep(20)
        else:
            raise RuntimeError('Monitoring deadline exceeded; do not relaunch')
        output=Path(tmp)/'output'; output.mkdir()
        run_cli('kernels','output',kernel,'-p',str(output),'--file-pattern','^metrics\\.json$')
        path=output/'metrics.json'
        if not path.is_file() or path.stat().st_size>16384:
            raise RuntimeError('Missing or oversized metrics; no publish')
        report=validate_metrics(json.loads(path.read_text()),run_id,sha)
        dest=Path('pilot-results'); dest.mkdir(exist_ok=True)
        (dest/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print('Validated WSP-PILOT-001 metrics:',json.dumps(report['metrics']),flush=True)


if __name__=='__main__':
    main()
