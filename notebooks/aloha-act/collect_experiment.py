"""Collect a cooled, measurement-phase-synced benchmark + power/thermal report.

Usage: python collect_experiment.py MODEL [LABEL] [PASSES] [SECONDS] [DEVICE]
"""

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import openvino as ov

REPO_DIR = Path(__file__).resolve().parent

MODEL = sys.argv[1] if len(sys.argv) > 1 else str(
    REPO_DIR / 'sim_insertion_scripted' / 'four_camera' / 'policy_best.xml'
)
LABEL = sys.argv[2] if len(sys.argv) > 2 else 'act_fp16'
PASSES = int(sys.argv[3]) if len(sys.argv) > 3 else 3
MEASURE_SECONDS = int(sys.argv[4]) if len(sys.argv) > 4 else 30
DEVICE = sys.argv[5] if len(sys.argv) > 5 else os.environ.get('DEVICE', 'GPU')
BENCH = os.environ.get('BENCHMARK_APP') or shutil.which('benchmark_app') or str(
    Path(sys.executable).with_name('benchmark_app')
)

GT0 = os.environ.get('GPU_FREQ_PATH', '/sys/devices/pci0000:00/0000:00:02.0/tile0/gt0/freq0')
PKG_TEMP = os.environ.get('PKG_TEMP_PATH', '/sys/class/hwmon/hwmon3/temp1_input')
RAPL = os.environ.get('RAPL_PATH', '/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0')
PERF_EVENTS = os.environ.get('PERF_EVENTS', (
    'uncore_imc_free_running/data_read/,'
    'uncore_imc_free_running/data_write/,'
    'xe_0000_00_02.0/engine-active-ticks/,'
    'xe_0000_00_02.0/engine-total-ticks/'
))

COOL_TEMP_C = 50.0
COOL_ACT_MHZ = 400
COOL_MAX_WAIT_S = 300


def read_int(path, default=None):
    try:
        return int(Path(path).read_text().strip())
    except Exception:
        return default


def gpu_act():
    return read_int(f'{GT0}/act_freq')


def gpu_max():
    return read_int(f'{GT0}/max_freq')


def gpu_pin(freq):
    """Lock GPU to a fixed frequency by setting min_freq == max_freq == freq."""
    Path(f'{GT0}/max_freq').write_text(str(freq))
    Path(f'{GT0}/min_freq').write_text(str(freq))


def gpu_unpin(orig_min, orig_max):
    Path(f'{GT0}/max_freq').write_text(str(orig_max))
    Path(f'{GT0}/min_freq').write_text(str(orig_min))


def pkg_temp():
    t = read_int(PKG_TEMP)
    return t / 1000.0 if t is not None else None


def memory_info():
    info = {'capacity_gb': None, 'speed_mtps': None, 'peak_gbps': None}
    try:
        out = subprocess.check_output(['dmidecode', '-t', 'memory'], text=True,
                                      stderr=subprocess.DEVNULL)
        capacities = [int(v) for v in re.findall(r'^\s*Size:\s+(\d+) GB$', out, re.MULTILINE)]
        widths = [int(v) for v in re.findall(r'^\s*Data Width:\s+(\d+) bits$', out, re.MULTILINE)]
        speeds = [int(v) for v in re.findall(r'^\s*Configured Memory Speed:\s+(\d+) MT/s$', out,
                                               re.MULTILINE)]
        if capacities:
            info['capacity_gb'] = sum(capacities)
        if speeds:
            info['speed_mtps'] = speeds[0]
        if widths and speeds:
            info['peak_gbps'] = sum(widths) / 8 * speeds[0] / 1000
    except Exception:
        pass
    return info


def start_perf_counters(duration_s):
    try:
        return subprocess.Popen(
            ['sudo', '-n', 'perf', 'stat', '-a', '-x,', '-e', PERF_EVENTS, '--', 'sleep', str(duration_s)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
    except Exception:
        return None


def stop_perf_counters(proc, duration_s):
    if proc is None:
        return {}
    _, out = proc.communicate()
    counts = {'duration_s': duration_s}
    for line in out.splitlines():
        fields = line.split(',')
        if len(fields) < 3:
            continue
        try:
            counts[fields[2]] = float(fields[0])
        except ValueError:
            continue
    return counts


def cool_gate():
    if pkg_temp() is None and gpu_act() is None:
        return 0.0, None, None
    start = time.time()
    while time.time() - start < COOL_MAX_WAIT_S:
        t, a = pkg_temp(), gpu_act()
        temp_ready = t is None or t <= COOL_TEMP_C
        gpu_ready = a is None or a <= COOL_ACT_MHZ
        if temp_ready and gpu_ready:
            break
        time.sleep(5)
    return round(time.time() - start, 1), pkg_temp(), gpu_act()


def run_pass(duration_s, sample=True):
    """Run benchmark_app; sample telemetry only during the measurement phase."""
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    proc = subprocess.Popen(
        [BENCH, '-m', MODEL, '-d', DEVICE, '-hint', 'latency', '-api', 'sync', '-t', str(duration_s)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
    )
    lines = []
    measuring = threading.Event()
    finished = threading.Event()

    def reader():
        for line in proc.stdout:
            lines.append(line)
            if 'Measuring performance' in line:
                measuring.set()
            elif 'Dumping statistics' in line:
                finished.set()
        measuring.set()
        finished.set()

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    samples = []
    perf_proc = None
    idle_start = None
    if sample:
        measuring.wait()
        idle_start = read_int(f'{GT0}/../gtidle/idle_residency_ms')
        perf_proc = start_perf_counters(duration_s)
        rng = read_int(f'{RAPL}/max_energy_range_uj', 10_000_000)
        prev_e, prev_t = read_int(f'{RAPL}/energy_uj'), time.time()
        while not finished.is_set() and proc.poll() is None:
            time.sleep(1.0)
            if finished.is_set() or proc.poll() is not None:
                break
            now = time.time()
            e = read_int(f'{RAPL}/energy_uj')
            power = None
            if e is not None and prev_e is not None:
                de = e - prev_e
                if de < 0:
                    de += rng
                power = (de / 1e6) / max(now - prev_t, 1e-6)
            samples.append({'power_w': power, 'gpu_mhz': gpu_act(), 'temp_c': pkg_temp()})
            prev_e, prev_t = e, now

    proc.wait()
    th.join(timeout=5)
    perf = stop_perf_counters(perf_proc, duration_s)
    idle_end = read_int(f'{GT0}/../gtidle/idle_residency_ms')
    if idle_start is not None and idle_end is not None:
        perf['gpu_idle_ms'] = max(idle_end - idle_start, 0)
    out = ''.join(lines)

    def grab(pattern, cast=float):
        m = re.search(pattern, out)
        return cast(m.group(1)) if m else None

    return {
        'returncode': proc.returncode,
        'median_ms': grab(r'Median:\s+([0-9.]+) ms'),
        'avg_ms': grab(r'Average:\s+([0-9.]+) ms'),
        'min_ms': grab(r'Min:\s+([0-9.]+) ms'),
        'max_ms': grab(r'Max:\s+([0-9.]+) ms'),
        'iterations': grab(r'Count:\s+([0-9]+) iterations', int),
        'fps': grab(r'Throughput:\s+([0-9.]+) FPS'),
        'samples': samples,
        'perf': perf,
        'output': out,
    }


def require_benchmark_metrics(result, phase):
    """Raise a useful error when benchmark_app fails or emits no measurements."""
    if result['returncode'] == 0 and result['median_ms'] is not None and result['fps'] is not None:
        return

    output_lines = [line.strip() for line in result['output'].splitlines() if line.strip()]
    details = '\n'.join(output_lines[-20:]) or 'benchmark_app produced no output'
    raise RuntimeError(
        f'benchmark_app could not complete the {phase} on device {DEVICE}. '
        f'Verify that the device is listed by OpenVINO and supports this model.\n{details}'
    )


def agg(samples, key):
    vals = [s[key] for s in samples if s[key] is not None]
    if not vals:
        return None, None, None
    return min(vals), statistics.mean(vals), max(vals)


def format_metric(value, spec):
    return format(value, spec) if value is not None else 'unavailable'


def main():
    if not Path(MODEL).is_file():
        raise FileNotFoundError(f'OpenVINO model not found: {MODEL}')
    if not Path(BENCH).is_file():
        raise FileNotFoundError(
            f'benchmark_app not found: {BENCH}. Set BENCHMARK_APP to its path on the target machine.'
        )

    available_devices = ov.Core().available_devices
    requested_device = DEVICE.split('.')[0].upper()
    available_device_types = {device.split('.')[0].upper() for device in available_devices}
    if requested_device not in available_device_types:
        raise RuntimeError(
            f'OpenVINO device {DEVICE!r} is unavailable. '
            f'Available devices: {", ".join(available_devices) or "none"}'
        )

    cpu = 'Unknown'
    try:
        m = re.search(r'Model name:\s*(.+)', subprocess.check_output(['lscpu'], text=True))
        if m:
            cpu = m.group(1).strip()
    except Exception:
        pass

    memory = memory_info()
    mem_speed = f"{memory['speed_mtps']} MT/s" if memory['speed_mtps'] else 'Unknown'
    rapl_available = read_int(f'{RAPL}/energy_uj') is not None
    perf_available = shutil.which('perf') is not None and shutil.which('sudo') is not None and subprocess.run(
        ['sudo', '-n', 'true'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0
    profiler_parts = ['power sampler' if rapl_available else 'power unavailable', 'thermal/frequency sampler']
    profiler_parts.append('perf IMC/XE counters' if perf_available else 'perf counters unavailable')
    profiler_text = ' + '.join(profiler_parts) + ' synced to measurement phase'

    pl1 = (read_int(f'{RAPL}/constraint_0_power_limit_uw') or 0) / 1e6
    pl2 = (read_int(f'{RAPL}/constraint_1_power_limit_uw') or 0) / 1e6
    gov = Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor').read_text().strip()

    wait_s, start_temp, start_mhz = cool_gate()
    print(f'[cool] waited {wait_s}s -> pkg={start_temp}C gpu={start_mhz}MHz', flush=True)

    gpu_selected = DEVICE.upper().startswith('GPU')
    req_freq = gpu_max() if gpu_selected else None
    orig_min = read_int(f'{GT0}/min_freq') if gpu_selected else None
    orig_max = read_int(f'{GT0}/max_freq') if gpu_selected else None
    gpu_pinned = gpu_selected and None not in (req_freq, orig_min, orig_max)
    if gpu_pinned:
        try:
            gpu_pin(req_freq)
            print(f'[gpu] pinned to {req_freq}MHz (cur={read_int(f"{GT0}/cur_freq")}MHz)', flush=True)
        except OSError as error:
            gpu_pinned = False
            print(f'[gpu] frequency pinning failed ({error}); continuing unpinned', flush=True)
    else:
        print('[gpu] frequency pinning unavailable or not applicable', flush=True)

    try:
        warmup = run_pass(10, sample=False)
        require_benchmark_metrics(warmup, 'warm-up')
        print('[warmup] done', flush=True)

        results = []
        for i in range(PASSES):
            r = run_pass(MEASURE_SECONDS)
            require_benchmark_metrics(r, f'pass {i + 1}')
            _, gmean, gmax = agg(r['samples'], 'gpu_mhz')
            results.append(r)
            gpu_mean_text = f'{gmean:.0f}MHz' if gmean is not None else 'unavailable'
            gpu_max_text = f'{gmax:.0f}MHz' if gmax is not None else 'unavailable'
            print(f"[pass {i+1}] median={r['median_ms']:.2f}ms fps={r['fps']:.2f} "
                  f"gpu_mean={gpu_mean_text} gpu_max={gpu_max_text}", flush=True)
    finally:
        if gpu_pinned:
            gpu_unpin(orig_min, orig_max)

    req = req_freq
    valid = [r for r in results if req is not None and agg(r['samples'], 'gpu_mhz')[1] and
             agg(r['samples'], 'gpu_mhz')[1] >= 0.90 * req]
    best = min(valid or results, key=lambda r: r['median_ms'])

    model_name = f'{LABEL} {MODEL}'.lower()
    if 'fp32' in model_name:
        precision, quant = 'FP32', 'None (dense FP32)'
    elif 'int4' in model_name:
        precision, quant = 'INT4', 'Symmetric, per-channel'
    elif 'int8' in model_name:
        precision, quant = 'INT8', 'NNCF static PTQ (weights + activations)'
    else:
        precision, quant = 'FP16', 'None (dense FP16)'

    camera_count = 4

    pmin, pmean, pmax = agg(best['samples'], 'power_w')
    gmin, gmean, gmaxv = agg(best['samples'], 'gpu_mhz') if gpu_selected else (None, None, None)
    tmin, tmean, tmaxv = agg(best['samples'], 'temp_c')
    perf = best['perf']
    mem_read = perf.get('uncore_imc_free_running/data_read/')
    mem_write = perf.get('uncore_imc_free_running/data_write/')
    mem_bandwidth = ((mem_read + mem_write) / perf['duration_s']
                     if mem_read is not None and mem_write is not None else None)
    mem_util = (mem_bandwidth / (memory['peak_gbps'] * 1_000 / 1.048576) * 100
                if mem_bandwidth is not None and memory['peak_gbps'] else None)
    gpu_idle_ms = perf.get('gpu_idle_ms')
    gpu_util = (1 - gpu_idle_ms / (perf.get('duration_s', 1) * 1000)) * 100 if gpu_selected and gpu_idle_ms is not None else None
    if gpu_util is not None:
        gpu_util = min(max(gpu_util, 0.0), 100.0)
    gpu_util_text = f'{gpu_util:.1f}' if gpu_util is not None else 'unavailable'
    mem_bandwidth_text = f'{mem_bandwidth:.1f}' if mem_bandwidth is not None else 'unavailable'
    mem_util_text = f'{mem_util:.2f}' if mem_util is not None else 'unavailable'
    throttle = ('No' if gmaxv >= req - 10 else 'Possible') if gmaxv is not None and req is not None else 'Unavailable'

    date = time.strftime('%Y-%m-%d')
    report = f"""Benchmarking Platform: Robinson Bay SKU (AAEON CEXD-INTRBL), {cpu} {mem_speed}, PL1={pl1:.0f}W, PL2={pl2:.0f}W
Model Source: ACT sim_insertion_scripted policy ({camera_count}-camera, {precision} OpenVINO IR)
Input: qpos [1,14] and camera tensor [1,{camera_count},3,480,640], static shapes
Output / Action Horizon: [1,100,14] / 100 actions
Conversion Method: PyTorch checkpoint to OpenVINO IR
Conditions: device={DEVICE}, governor={gov}, fans=auto, GPU freq pinned={format_metric(req_freq, '.0f') + 'MHz' if gpu_pinned else 'No'}
Profiler: {profiler_text}
Benchmark Date: {date}

Performance Hint\tPrecision\tQuantization\tMedian Latency (ms)\tFPS
Latency\t{precision}\t{quant}\t{best['median_ms']:.2f}\t{best['fps']:.2f}

Latency Detail\tMedian (ms)\tMin (ms)\tMax (ms)\tIterations
\t{best['median_ms']:.2f}\t{best['min_ms']:.2f}\t{best['max_ms']:.2f}\t{best['iterations']}

Edge / Power Metric\tMin\tMean\tMax
Package Power / TDP (W)\t{format_metric(pmin, '.1f')}\t{format_metric(pmean, '.1f')}\t{format_metric(pmax, '.1f')}
GPU Act Frequency (MHz)\t{format_metric(gmin, '.0f')}\t{format_metric(gmean, '.1f')}\t{format_metric(gmaxv, '.0f')}
GPU Engine Utilization (%)\t\t{gpu_util_text}\t
Package Temp (°C)\t{format_metric(tmin, '.0f')}\t{format_metric(tmean, '.1f')}\t{format_metric(tmaxv, '.0f')}
GPU Throttle\t{throttle} ({format_metric(req, '.0f')} MHz requested throughout)
Memory Capacity (GB)\t{memory['capacity_gb'] if memory['capacity_gb'] else 'unavailable'}
Memory Bandwidth (MiB/s)\t\t{mem_bandwidth_text}\t
Memory Bandwidth Utilization (%)\t\t{mem_util_text}\t
"""
    out_txt = REPO_DIR / f'experiment_{LABEL}_{date}.txt'
    out_txt.write_text(report)
    (REPO_DIR / f'experiment_{LABEL}_{date}.json').write_text(
        json.dumps({'cool_wait_s': wait_s, 'start_temp_c': start_temp,
                    'passes': results, 'requested_mhz': req}, indent=2))
    print('\n' + report)
    print(f'Saved: {out_txt}')


if __name__ == '__main__':
    main()
