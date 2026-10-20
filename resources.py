"""Admission and allocator cap for a shared GPU; never kill or modify other jobs."""
import json
import hashlib
import fcntl
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/data3/dzh/MedSearchRL')


def lock_run(path):
    """Keep the returned file open for the entire process to reserve its output directory."""
    handle = (path / '.run.lock').open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f'Another process owns run directory {path}') from None
    return handle


def read_jsonl(path, repair_tail=False):
    """Recover only a final interrupted append; internal corruption always fails."""
    original = path.read_bytes()
    lines = original.splitlines(keepends=True)
    result = []
    repaired = False
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if not repair_tail or index != len(lines) - 1 or line.endswith(b'\n'):
                raise
            repaired = True
    if repair_tail and lines and (repaired or not lines[-1].endswith(b'\n')):
        path.with_name(path.name + f'.before-repair-{time.time_ns()}').write_bytes(original)
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in result))
    return result


def environment_hash():
    base = Path(__file__).parent
    return hashlib.sha256(b''.join((base / name).read_bytes() for name in ('evisearch/environment.py', 'evisearch/model.py'))).hexdigest()


def setup_paths():
    for key, suffix in {'HF_HOME': 'cache/huggingface', 'TMPDIR': 'tmp',
                        'TRITON_CACHE_DIR': 'cache/triton', 'TORCHINDUCTOR_CACHE_DIR': 'cache/inductor',
                        'TORCH_HOME': 'cache/torch', 'XDG_CACHE_HOME': 'cache/xdg'}.items():
        path = ROOT / suffix
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'


def gpu_status():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.total,memory.free,utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return [dict(zip(['index', 'total_mib', 'free_mib', 'utilization'], map(int, line.split(','))))
            for line in result.stdout.strip().splitlines()]


def admit(gpu=0, cap_gib=12, headroom_gib=6, wait=True):
    setup_paths()
    while True:
        state = next(x for x in gpu_status() if x['index'] == gpu)
        if state['free_mib'] >= (cap_gib + headroom_gib) * 1024:
            break
        print(json.dumps({'event': 'waiting_for_gpu_memory', 'gpu': state,
                          'required_free_mib': (cap_gib + headroom_gib) * 1024}), flush=True)
        if not wait:
            raise RuntimeError('Insufficient free GPU memory; no allocation attempted')
        time.sleep(30)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import torch
    torch.set_num_threads(4)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(cap_gib * 1024 ** 3 / total, device=0)
    torch.cuda.reset_peak_memory_stats()
    print(json.dumps({'event': 'gpu_admitted', 'gpu': state, 'allocator_cap_gib': cap_gib,
                      'minimum_headroom_gib': headroom_gib}), flush=True)
    return state


def memory_record(gpu=0):
    import torch
    state = next(x for x in gpu_status() if x['index'] == gpu)
    return dict(state, allocated_gib=torch.cuda.memory_allocated() / 2 ** 30,
                reserved_gib=torch.cuda.memory_reserved() / 2 ** 30,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30)


def admit_pair(gpu, cap_gib, teacher_gpu, teacher_cap_gib, headroom_gib=6):
    """Admit one experiment that places a frozen teacher on a second GPU (cuda:1)."""
    setup_paths()
    while True:
        status = {x['index']: x for x in gpu_status()}
        if (status[gpu]['free_mib'] >= (cap_gib + headroom_gib) * 1024
                and status[teacher_gpu]['free_mib'] >= (teacher_cap_gib + headroom_gib) * 1024):
            break
        print(json.dumps({'event': 'waiting_for_gpu_memory', 'gpus': [status[gpu], status[teacher_gpu]]}), flush=True)
        time.sleep(30)
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu},{teacher_gpu}'
    import torch
    torch.set_num_threads(4)
    for device, cap in ((0, cap_gib), (1, teacher_cap_gib)):
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(cap * 1024 ** 3 / total, device=device)
    torch.cuda.reset_peak_memory_stats()
    print(json.dumps({'event': 'gpu_pair_admitted', 'student_gpu': gpu, 'teacher_gpu': teacher_gpu,
                      'caps_gib': [cap_gib, teacher_cap_gib], 'minimum_headroom_gib': headroom_gib}), flush=True)
