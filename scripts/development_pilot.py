#!/usr/bin/env python3
"""Detached, immutable-source development-pilot launcher (Linux/macOS)."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

SOURCE = Path(__file__).resolve().parents[1]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text())


def save(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, p)


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def ref(p):
    p = Path(p).resolve()
    return {'path': str(p), 'sha256': digest(p), 'bytes': p.stat().st_size}


def checked(r):
    p = Path(r['path'])
    if p.stat().st_size != r['bytes'] or digest(p) != r['sha256']:
        raise ValueError('Artifact changed: '+str(p))
    return p


def event(root, **row):
    with (root/'launcher_events.jsonl').open('a') as f:
        f.write(json.dumps({'time': now(), **row})+'\n'); f.flush(); os.fsync(f.fileno())


def identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return stat[19]  # /proc starttime, excludes PID reuse
    except FileNotFoundError:
        p = subprocess.run(['ps', '-p', str(pid), '-o', 'lstart='], capture_output=True, text=True)
        return p.stdout.strip() or None


def alive(record):
    return bool(record and record.get('identity') and identity(record['pid']) == record['identity'])


def verify_snapshot(root):
    manifest = read(root/'launch_manifest.json')
    checked(manifest['plan'])
    for entry in manifest['source_files']:
        checked({'path': str(root/'source'/entry['path']), 'bytes': entry['bytes'], 'sha256': entry['sha256']})
    # Reject additional Python files as well as modifications of recorded files.
    expected = {x['path'] for x in manifest['source_files'] if x['path'].endswith('.py')}
    actual = {str(p.relative_to(root/'source')) for p in (root/'source').rglob('*.py')}
    if expected != actual:
        raise ValueError('Snapshot Python file set changed')
    return manifest


def setup(args):
    root = args.root.resolve()
    if root == SOURCE or SOURCE in root.parents:
        raise ValueError('Use a study directory outside the Git checkout')
    plan = read(args.config)
    root.mkdir(parents=True, exist_ok=False)
    save(root/'plan.json', plan)
    entries = []
    for directory in ('dpnn_experiments', 'src', 'scripts', 'configs', 'tests', 'tools'):
        for p in sorted((SOURCE/directory).rglob('*')):
            if p.is_file() and p.suffix in ('.py', '.json', '.sh') and '__pycache__' not in p.parts:
                if p.is_symlink():
                    raise ValueError('Source symlinks are not supported')
                relative = p.relative_to(SOURCE)
                dest = root/'source'/relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
                entries.append({'path': str(relative), 'sha256': digest(dest), 'bytes': dest.stat().st_size})
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, capture_output=True, text=True)
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=SOURCE, capture_output=True, text=True)
    # Only record execution-relevant environment variables, never the complete environment.
    keys = ('CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG', 'PYTHONHASHSEED')
    environment = {k: os.environ.get(k) for k in keys}
    save(root/'launch_manifest.json', {'schema_version': 1, 'created': now(), 'python': sys.executable,
        'python_version': sys.version, 'git_commit': git.stdout.strip(), 'git_status': dirty.stdout,
        'plan': ref(root/'plan.json'), 'source_files': entries, 'environment': environment,
        'retention': 'Keep all attempts, stage archives, checkpoints, caches and dependencies',
        'authorization': 'Explicit command-line launch; no scientific test evaluation'})
    return root


def launch(root, foreground=False):
    identity(os.getpid())  # Fail before spawning if process inspection is unavailable.
    manifest = verify_snapshot(root)
    lock = (root/'worker.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Pilot worker is already active')
    state = read(root/'status.json') if (root/'status.json').exists() else {}
    if alive(state.get('child')):
        raise RuntimeError('Earlier stage is still active; do not launch a second copy')
    if state.get('status') == 'completed':
        raise RuntimeError('Pilot is already completed; use a new directory for a new study')
    for k, v in manifest['environment'].items():
        if os.environ.get(k) != v:
            raise RuntimeError('Execution environment differs from launch: '+k)
    if sys.executable != manifest['python']:
        raise RuntimeError('Use the same Python executable as the original launch')
    (root/'STOP_REQUESTED').unlink(missing_ok=True)
    command = [manifest['python'], '-u', str(root/'source/scripts/development_pilot.py'),
               '_worker', '--root', str(root), '--lock-fd', str(lock.fileno())]
    env = os.environ.copy(); env['PYTHONPATH'] = str(root/'source'); env['PYTHONUNBUFFERED'] = '1'
    with (root/'console.log').open('ab', buffering=0) as log:
        p = subprocess.Popen(command, cwd=root/'source', env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=not foreground, pass_fds=(lock.fileno(),))
    # Child inherits the locked descriptor; closing here does not release its lock.
    lock.close()
    save(root/'worker.json', {'pid': p.pid, 'identity': identity(p.pid), 'started': now()})
    event(root, event='worker_launched', pid=p.pid, foreground=foreground)
    print(f'Pilot directory: {root}\nPID: {p.pid}\nLog: {root / "console.log"}', flush=True)
    if foreground:
        try:
            return p.wait()
        except KeyboardInterrupt:
            (root/'STOP_REQUESTED').touch()
            return p.wait()
    time.sleep(.3)
    if p.poll() is not None and p.returncode:
        raise RuntimeError('Worker failed during startup; see console.log')
    return 0


def worker(root, fd):
    # Inherited FD keeps an exclusive study lock through all stages.
    os.fstat(fd)
    manifest = verify_snapshot(root)
    plan = read(root/'plan.json')
    sys.path.insert(0, str(root/'source'))
    stopped = {'value': False}
    child = {'process': None}
    def stop(sig, frame):
        stopped['value'] = True
        (root/'STOP_REQUESTED').touch()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def update(status, **kw):
        save(root/'status.json', {'status': status, 'updated': now(), 'worker_pid': os.getpid(), **kw})
    def run_process(command, log, label):
        if stopped['value'] or (root/'STOP_REQUESTED').exists():
            raise InterruptedError('Stop requested between stages')
        with log.open('ab', buffering=0) as out:
            p = subprocess.Popen(command, cwd=root/'source', stdout=out, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        child['process'] = p
        update('running', stage=label, child={'pid': p.pid, 'identity': identity(p.pid)})
        event(root, event='stage_process_started', stage=label, pid=p.pid, command=command, log=str(log))
        sent = False
        print(f'{now()} START {label}: {log}', flush=True)
        last_sample = 0.0
        began = time.monotonic()
        while p.poll() is None:
            if time.monotonic() - last_sample >= 30:
                last_sample = time.monotonic()
                print(f'{now()} RUNNING {label}, elapsed {last_sample-began:.0f}s', flush=True)
                if plan['device'].startswith('cuda'):
                    try:
                        sample = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True, timeout=5)
                        event(root, event='gpu_sample', stage=label, returncode=sample.returncode, stdout=sample.stdout, stderr=sample.stderr)
                    except (OSError, subprocess.TimeoutExpired) as error:
                        event(root, event='gpu_sample_unavailable', reason=str(error))
            if stopped['value'] or (root/'STOP_REQUESTED').exists():
                stopped['value'] = True
                if not sent:
                    p.send_signal(signal.SIGTERM); sent = True
                    event(root, event='stage_stop_sent', stage=label)
            time.sleep(.25)
        child['process'] = None
        print(f'{now()} END {label}: exit {p.returncode}', flush=True)
        event(root, event='stage_process_finished', stage=label, returncode=p.returncode)
        if stopped['value'] or p.returncode == 75:
            raise InterruptedError('Stage interrupted; checkpoint if available is retained')
        if p.returncode:
            raise RuntimeError(f'{label} failed ({p.returncode}); see {log}')
    def check_result(result):
        run = Path(result['run'])
        if read(run/'completion.json')['status'] != 'completed':
            raise ValueError('Previously completed stage is no longer completed')
        checked(result['completion']); checked(result['artifacts'])
        index = read(run/'artifacts.json')['files']
        for item in index:
            checked({'path': str(run/item['path']), 'bytes': item['bytes'], 'sha256': item['sha256']})
        return run
    def run_stage(label, kind, config, operator_root):
        if stopped['value'] or (root/'STOP_REQUESTED').exists():
            raise InterruptedError('Stop requested between stages')
        if shutil.disk_usage(root).free < plan.get('stage_free_reserve_gib', 20)*1024**3:
            raise RuntimeError('Free disk space below stage reserve; previous artifacts retained')
        folder = root/'stages'/label; folder.mkdir(parents=True, exist_ok=True)
        intended = {'kind': kind, 'config': config, 'operator_root': operator_root}
        if (folder/'intent.json').exists():
            if read(folder/'intent.json') != intended:
                raise ValueError('Stage configuration changed: '+label)
        else:
            save(folder/'intent.json', intended)
        attempts = sorted(folder.glob('attempt_*'))
        for previous in reversed(attempts):
            result_path = previous/'result.json'
            if result_path.exists() and read(result_path)['status'] == 'completed':
                return check_result(read(result_path))
            # Recover a completed archive after a crash before publishing result.json.
            runs = sorted((previous/'runs').glob('*'))
            for run in reversed(runs):
                if (run/'completion.json').exists() and read(run/'completion.json')['status'] == 'completed':
                    result = {'status': 'completed', 'run': str(run),
                        'completion': ref(run/'completion.json'), 'artifacts': ref(run/'artifacts.json')}
                    save(result_path, result)
                    return check_result(result)
        attempt = folder/f'attempt_{len(attempts)+1:04d}'; attempt.mkdir()
        actual = json.loads(json.dumps(config))
        if kind == 'train':
            checkpoints = [p for a in attempts for p in (a/'runs').glob('*/latest_checkpoint.json')]
            if checkpoints:
                cp = sorted(checkpoints, key=lambda p: p.stat().st_mtime_ns)[-1]
                actual['execution']['resume'] = read(cp)
                checked(actual['execution']['resume'])
        spec = {**intended, 'config': actual, 'root': str(attempt), 'result': str(attempt/'result.json')}
        save(attempt/'stage.json', spec)
        event(root, event='stage_attempt', stage=label, attempt=str(attempt), configuration=ref(attempt/'stage.json'))
        run_process([manifest['python'], '-u', '-m', 'dpnn_experiments.development_pilot',
                     '--stage', str(attempt/'stage.json')], attempt/'process.log', label)
        result = read(attempt/'result.json'); run = Path(result['run'])
        result.update(completion=ref(run/'completion.json'), artifacts=ref(run/'artifacts.json'))
        save(attempt/'result.json', result)
        return check_result(result)
    try:
        update('preflight')
        required = plan['minimum_free_gib']*1024**3
        if shutil.disk_usage(root).free < required:
            raise RuntimeError(f'Less than {plan["minimum_free_gib"]} GiB free; no stage started')
        pre = root/('preflight_'+uuid.uuid4().hex[:12]); pre.mkdir()
        save(pre/'stage.json', {'kind': 'preflight', 'config': plan, 'result': str(pre/'result.json')})
        run_process([manifest['python'], '-u', '-m', 'dpnn_experiments.development_pilot',
                     '--stage', str(pre/'stage.json')], pre/'process.log', 'preflight')
        from dpnn_experiments.development_pilot import workflow
        result = workflow(plan, root, run_stage)
        save(root/'summary.json', result)
        update('completed', summary=str(root/'summary.json'))
        event(root, event='completed')
        return 0
    except InterruptedError as e:
        update('interrupted', reason=str(e)); event(root, event='interrupted', reason=str(e)); return 75
    except BaseException as e:
        traceback.print_exc()
        update('failed', reason=str(e)); event(root, event='failed', reason=str(e)); return 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['start', 'run', 'resume', 'status', 'stop', '_worker'])
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--config', type=Path, default=SOURCE/'configs/study/development_pilot_256.json')
    p.add_argument('--lock-fd', type=int)
    p.add_argument('--foreground', action='store_true', help='Wait for the resumed worker (controller use)')
    args = p.parse_args(); root = args.root.resolve()
    if args.action in ('start', 'run'):
        root = setup(args)
        return launch(root, foreground=args.action == 'run')
    if args.action == 'resume':
        return launch(root, foreground=args.foreground)
    if args.action == '_worker':
        return worker(root, args.lock_fd)
    if args.action == 'stop':
        if not (root/'launch_manifest.json').exists():
            raise ValueError('Not a pilot directory')
        (root/'STOP_REQUESTED').touch()
        event(root, event='user_stop_requested')
        print('Stop requested. Training saves at the next safe optimizer-step boundary; wait for interrupted status.')
        return 0
    state = read(root/'status.json') if (root/'status.json').exists() else {'status': 'not_started'}
    process = read(root/'worker.json') if (root/'worker.json').exists() else None
    state['worker_alive'] = alive(process)
    if not state['worker_alive'] and state['status'] in ('running', 'preflight'):
        state['status'] = 'worker_lost'; state['advice'] = 'Check orphan stage and logs before resume'
    print(json.dumps(state, indent=2)); return 0


if __name__ == '__main__':
    sys.exit(main())
