#!/usr/bin/env python3
# Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

"""Tune MFEM's native batched gemv GPU kernels for the GPU of this machine.

The kernels (linalg/batched/native_gemv.hpp) compute y_e = op(A_e) x_e for
many small matrices A_e, e.g. in element-assembled operators. Their tuning
parameters have defaults for NVIDIA and AMD GPUs; this script measures the
candidate values on the current GPU and writes a header with the best ones.

Steps:
  1. Build MFEM with CUDA or HIP (CMake or make, installed or not).
  2. On a machine (or compute node) with the target GPU, run
        python3 config/tune_batched_gemv.py --mfem-dir <MFEM build or install>
     It compiles config/tune_batched_gemv.cpp with the flags of that MFEM
     build (from its config.mk), runs it, and writes the tuning header.
  3. Rebuild MFEM with the header:
        CMake: cmake -DMFEM_BATCHED_GEMV_TUNING=<header> <build dir>
               (or run this script with --apply-cmake <build dir>)
        make:  make config MFEM_BATCHED_GEMV_TUNING=<header> <other options>
               (or add the setting to config/user.mk)
     then rebuild and reinstall MFEM.

Requires Python 3.6+ and GNU make (to read config.mk). Matplotlib is used only
by the optional --plot.
"""

import argparse
import concurrent.futures
import datetime
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MFEM_SRC = os.path.dirname(HERE)
BENCH_SRC = os.path.join(HERE, 'tune_batched_gemv.cpp')
KERNEL_DIR = os.path.join(MFEM_SRC, 'linalg', 'batched')

# Element sizes (dofs per element) of low-order (p <= 2) scalar and vector H1
# and L2 spaces on 1D/2D/3D elements; weighted more when choosing parameters.
FE_SIZES = [1, 2, 3, 4, 6, 8, 9, 10, 12, 18, 24, 27, 30, 54, 81]

PARAMS = ['SMALL_BLOCK_N', 'SMALL_BLOCK_T', 'SMALL_T_MAX', 'MED_N_ROUND',
          'MED_N_BLOCK', 'MED_T_G1', 'MED_T_NT1', 'MED_T_SWITCH', 'MED_T_G2',
          'MED_T_NT2']

# Candidate values. Small blocks are whole warps (32) or wavefronts (64).
SMALL_BLOCKS = {'cuda': [32, 64, 128, 256], 'hip': [64, 128, 256, 512]}
MEDT_CONFIGS = [(g, nt) for g in (4, 8, 16, 32) for nt in (64, 128, 256, 512)]
MEDN_CONFIGS = [(r, b) for r in (32, 64) for b in (128, 256, 512)]


def log(msg):
    print(msg, flush=True)


def die(msg):
    sys.exit('tune_batched_gemv.py: error: ' + msg)


def find_config_mk(mfem_dir):
    for rel in ('config/config.mk', 'share/mfem/config.mk', 'config.mk'):
        p = os.path.join(mfem_dir, rel)
        if os.path.isfile(p):
            return p
    die('no config.mk found under %s (expected config/config.mk in a build '
        'directory or share/mfem/config.mk in an install directory)' % mfem_dir)


def make_vars(config_mk, names, make='make'):
    """Values of the given variables after GNU make expansion of config.mk."""
    with tempfile.NamedTemporaryFile('w', suffix='.mk', delete=False) as f:
        f.write('include %s\n' % config_mk)
        for v in names:
            f.write('$(info @@%s@@$(%s))\n' % (v, v))
        f.write('all: ; @:\n')
        tmp = f.name
    try:
        out = subprocess.run([make, '-s', '-f', tmp, 'all'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        die('could not read %s with make: %s' % (config_mk, e))
    finally:
        os.unlink(tmp)
    vals = {}
    for line in out.splitlines():
        m = re.match(r'@@(\w+)@@(.*)$', line)
        if m:
            vals[m.group(1)] = m.group(2).strip()
    return vals


def xlist(configs):
    # One number per configuration: nvcc splits -D values at commas.
    return ''.join('X(%d)' % (c[0] if len(c) == 1 else 1000 * c[0] + c[1])
                   for c in configs)


def build(args, backend, mk):
    """Compile the benchmark in parallel parts; return the executable path."""
    os.makedirs(args.work_dir, exist_ok=True)
    exe = os.path.join(args.work_dir, 'tune_batched_gemv')
    common = (shlex.split(mk['MFEM_FLAGS']) + shlex.split(args.extra_cxxflags) +
              ['-I' + KERNEL_DIR])
    common += ['-DTUNE_SMALL_LIST=' + xlist([(b,) for b in SMALL_BLOCKS[backend]]),
               '-DTUNE_MEDT_LIST=' + xlist(MEDT_CONFIGS),
               '-DTUNE_MEDN_LIST=' + xlist(MEDN_CONFIGS)]
    parts = [('main', ['-DTUNE_MAIN'])]
    parts += [('small%d' % b, ['-DTUNE_INST_SMALL=%d' % b])
              for b in SMALL_BLOCKS[backend]]
    for i in range(0, len(MEDT_CONFIGS), 4):
        parts.append(('medt%d' % i,
                      ['-DTUNE_INST_MEDT=' + xlist(MEDT_CONFIGS[i:i + 4])]))
    parts.append(('medn', ['-DTUNE_INST_MEDN=' + xlist(MEDN_CONFIGS)]))

    cxx = shlex.split(mk['MFEM_CXX'])

    def compile_part(part):
        name, defs = part
        obj = os.path.join(args.work_dir, name + '.o')
        cmd = cxx + common + defs + ['-c', BENCH_SRC, '-o', obj]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True)
        with open(os.path.join(args.work_dir, name + '.log'), 'w') as f:
            f.write(' '.join(shlex.quote(c) for c in cmd) + '\n' + r.stdout)
        return name, obj, r.returncode, r.stdout

    jobs = args.jobs or min(len(parts), os.cpu_count() or 1)
    log('Compiling the benchmark (%d parts, %d at a time; this takes a few '
        'minutes) in %s' % (len(parts), jobs, args.work_dir))
    objs = []
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        for name, obj, rc, out in pool.map(compile_part, parts):
            if rc != 0:
                die('compiling part %s failed:\n%s' % (name, out[-4000:]))
            log('  compiled %s' % name)
            objs.append(obj)
    cmd = (cxx + shlex.split(mk['MFEM_LINK_FLAGS']) + objs + ['-o', exe] +
           shlex.split(mk['MFEM_LIBS']) + shlex.split(args.extra_libs))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True)
    if r.returncode != 0:
        die('linking failed:\n%s\n%s' % (' '.join(cmd), r.stdout[-4000:]))
    return exe


def run(args, exe):
    cmd = shlex.split(args.launcher) + [
        exe, str(args.budget_mb), str(args.trial_ms),
        '0' if args.no_vendor else '1', str(args.stride)]
    log('Running: ' + ' '.join(shlex.quote(c) for c in cmd))
    lines = []
    raw = os.path.join(args.work_dir, 'results.txt')
    with open(raw, 'w') as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
        last = None
        for line in p.stdout:
            f.write(line)
            f.flush()
            lines.append(line)
            w = line.split()
            if w and w[0] == 'R' and (w[3], w[4]) != last:
                last = (w[3], w[4])
                if int(w[4]) % 16 == 1 or int(w[4]) == 128:
                    log('  op %s, n = %s' % last)
            elif w and w[0] == 'INFO' and w[1] in ('name', 'arch'):
                log('  GPU %s: %s' % (w[1], ' '.join(w[2:])))
        if p.wait() != 0:
            die('the benchmark failed (exit code %d), see %s' % (p.returncode, raw))
    log('Raw results: ' + raw)
    return lines


class Results:
    def __init__(self, lines):
        self.info = {}
        self.default = {}
        self.r = {}       # (family, cfg, op, n) -> GB/s
        self.bad = []
        for line in lines:
            w = line.split()
            if not w:
                continue
            if w[0] == 'INFO':
                self.info[w[1]] = ' '.join(w[2:])
            elif w[0] == 'DEFAULT':
                self.default = {k: int(v) for k, v in
                                (kv.split('=') for kv in w[1:])}
        tol_unit = 1e-12 if self.info.get('real_t_bytes', '8') == '8' else 1e-4
        for line in lines:
            w = line.split()
            if not w or w[0] != 'R':
                continue
            fam, cfg, op, n, gbs, err = w[1], w[2], w[3], int(w[4]), float(w[7]), float(w[8])
            if not err <= tol_unit * max(1, n) * 10:
                self.bad.append((fam, cfg, op, n, err))
                continue
            self.r[(fam, cfg, op, n)] = gbs
        if 'done' not in self.info:
            die('incomplete benchmark output')
        if not self.default:
            die('no DEFAULT line in the benchmark output')
        # Best native result per (op, n), over all native kernels.
        self.best = {}
        for (fam, cfg, op, n), g in self.r.items():
            if fam != 'vendor':
                self.best[(op, n)] = max(self.best.get((op, n), 0.0), g)

    def get(self, fam, cfg, op, n):
        return self.r.get((fam, cfg, op, n))


def cfg_str(*v):
    return '/'.join(str(x) for x in v)


def predict(res, p, op, n):
    """GB/s of the kernel that the parameters p choose for an n x n matrix."""
    if op == 'T' and (n > 32 or n > p['SMALL_T_MAX']):
        if n <= p['MED_T_SWITCH']:
            return res.get('medt', cfg_str(p['MED_T_G1'], p['MED_T_NT1']), op, n)
        return res.get('medt', cfg_str(p['MED_T_G2'], p['MED_T_NT2']), op, n)
    if n > 32:
        return res.get('medn', cfg_str(p['MED_N_ROUND'], p['MED_N_BLOCK']), op, n)
    return res.get('small', str(p['SMALL_BLOCK_T' if op == 'T' else 'SMALL_BLOCK_N']),
                   op, n)


def score(res, fe_weight, op, sizes, gbs_of):
    """Weighted mean over sizes of GB/s relative to the best native kernel;
    None if a size has no (valid) measurement."""
    tot = wsum = 0.0
    for n in sizes:
        g = gbs_of(n)
        if g is None:
            return None
        w = fe_weight if n in FE_SIZES else 1.0
        tot += w * g / res.best[(op, n)]
        wsum += w
    return tot / wsum if wsum else None


def choose(cands, default, tol=0.002):
    """cands: list of (score, value). Keep the default unless another value
    scores better by more than tol (to avoid following timing noise)."""
    cands = [c for c in cands if c[0] is not None]
    if not cands:
        return default, None
    best = max(cands, key=lambda c: c[0])
    for s, v in cands:
        if v == default and s >= best[0] - tol:
            return v, s
    return best[1], best[0]


def tune(res, fe_weight):
    d = res.default
    p = dict(d)
    sizes = lambda op, fam, lo, hi: sorted(
        n for (f, c, o, n) in res.r if f == fam and o == op and lo <= n <= hi)
    small_n = sorted(set(sizes('N', 'small', 1, 32)))
    blocks = sorted(set(int(c) for (f, c, o, n) in res.r if f == 'small'))

    # Small NoTrans: block size.
    p['SMALL_BLOCK_N'], _ = choose(
        [(score(res, fe_weight, 'N', small_n,
                lambda n, b=b: res.get('small', str(b), 'N', n)), b)
         for b in blocks], d['SMALL_BLOCK_N'])

    # Medium NoTrans: (round, block).
    med_n = sorted(set(sizes('N', 'medn', 33, 128)))
    rb, _ = choose(
        [(score(res, fe_weight, 'N', med_n,
                lambda n, c=c: res.get('medn', cfg_str(*c), 'N', n)), c)
         for c in MEDN_CONFIGS],
        (d['MED_N_ROUND'], d['MED_N_BLOCK']))
    p['MED_N_ROUND'], p['MED_N_BLOCK'] = rb

    # Medium Trans: two configurations and the size where they switch.
    med_t = sorted(set(sizes('T', 'medt', 33, 128)))
    cands = []
    for c1 in MEDT_CONFIGS:
        for c2 in MEDT_CONFIGS:
            if c1 == c2:
                switches = [128]
            else:
                switches = [s for s in med_t if s < 128]
            for s in switches:
                q = dict(p, MED_T_G1=c1[0], MED_T_NT1=c1[1], MED_T_SWITCH=s,
                         MED_T_G2=c2[0], MED_T_NT2=c2[1], SMALL_T_MAX=32)
                cands.append((score(res, fe_weight, 'T', med_t,
                                    lambda n, q=q: predict(res, q, 'T', n)),
                              (c1, s, c2)))
    dflt = ((d['MED_T_G1'], d['MED_T_NT1']), d['MED_T_SWITCH'],
            (d['MED_T_G2'], d['MED_T_NT2']))
    if dflt not in [c[1] for c in cands]:
        q = dict(p, MED_T_G1=dflt[0][0], MED_T_NT1=dflt[0][1],
                 MED_T_SWITCH=dflt[1], MED_T_G2=dflt[2][0],
                 MED_T_NT2=dflt[2][1], SMALL_T_MAX=32)
        cands.append((score(res, fe_weight, 'T', med_t,
                            lambda n: predict(res, q, 'T', n)), dflt))
    (c1, s, c2), _ = choose(cands, dflt)
    if c1 == c2:
        s = 128
    p['MED_T_G1'], p['MED_T_NT1'] = c1
    p['MED_T_SWITCH'] = s
    p['MED_T_G2'], p['MED_T_NT2'] = c2

    # Small Trans: block size, and the largest size kept on the small kernel
    # (larger ones use the medium Trans kernel).
    small_t = sorted(set(sizes('T', 'small', 1, 32)))
    cands = []
    for b in blocks:
        for tmax in range(8, 33):
            q = dict(p, SMALL_BLOCK_T=b, SMALL_T_MAX=tmax)
            cands.append((score(res, fe_weight, 'T', small_t,
                                lambda n, q=q: predict(res, q, 'T', n)),
                          (b, tmax)))
    (p['SMALL_BLOCK_T'], p['SMALL_T_MAX']), _ = choose(
        cands, (d['SMALL_BLOCK_T'], d['SMALL_T_MAX']))
    return p


def summary_rows(res, tuned):
    rows = []
    vend = sorted(set(c for (f, c, o, n) in res.r if f == 'vendor'))
    for op in ('N', 'T'):
        for n in range(1, 129):
            t = predict(res, tuned, op, n)
            d = predict(res, res.default, op, n)
            if t is None or d is None:
                continue
            v = {c: res.get('vendor', c, op, n) for c in vend}
            rows.append((op, n, t, d, res.best[(op, n)], v))
    return rows, vend


def write_header(path, res, tuned, rows, vend, backend, args):
    info = res.info
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    L = []
    L.append('// Tuned parameters for MFEM\'s native batched gemv GPU kernels')
    L.append('// (linalg/batched/native_gemv.hpp), generated by')
    L.append('// config/tune_batched_gemv.py on %s.' % now)
    L.append('//')
    L.append('// GPU: %s (%s), %s compute units' % (
        info.get('name', '?'), info.get('arch', '?'), info.get('compute_units', '?')))
    L.append('// MFEM: %s' % os.path.abspath(args.mfem_dir))
    L.append('// Benchmark: %s MB of A + x + y per test, fp%d.' % (
        info.get('budget_mb', '?'), 8 * int(info.get('real_t_bytes', '8'))))
    L.append('//')
    L.append('// Use: cmake -DMFEM_BATCHED_GEMV_TUNING=<this file> ..., or')
    L.append('//      make config MFEM_BATCHED_GEMV_TUNING=<this file> ...')
    L.append('')
    L.append('#define MFEM_GEMV_TUNED_FOR_HIP %d' % (backend == 'hip'))
    L.append('')
    for k in PARAMS:
        note = '' if tuned[k] == res.default[k] else \
               '  // default %d' % res.default[k]
        L.append('#define MFEM_GEMV_%s %d%s' % (k, tuned[k], note))
    L.append('')
    L.append('// Measured GB/s (A + x + y): tuned, default parameters, and the')
    L.append('// vendor libraries through BatchedLinAlg (where measured).')
    hdr = '//  op    n   tuned  default' + ''.join(' %9s' % c for c in vend)
    L.append(hdr)
    for op, n, t, d, b, v in rows:
        if n in FE_SIZES or n % 16 == 0:
            L.append('//  %s  %4d  %6.1f  %7.1f' % (op, n, t, d) + ''.join(
                ' %9s' % ('%.1f' % v[c] if v[c] is not None else '-') for c in vend))
    L.append('')
    with open(path, 'w') as f:
        f.write('\n'.join(L))


def plot(path, rows, vend, info):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        log('matplotlib not found, skipping --plot')
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for ax, op in zip(axes, ('N', 'T')):
        r = [x for x in rows if x[0] == op]
        ax.plot([x[1] for x in r], [x[2] for x in r], '-', label='native, tuned')
        ax.plot([x[1] for x in r], [x[3] for x in r], '--', label='native, default')
        for c in vend:
            rv = [(x[1], x[5][c]) for x in r if x[5][c] is not None]
            ax.plot([a for a, _ in rv], [g for _, g in rv], 'o-', ms=3, label=c)
        ax.set_title('op %s' % op)
        ax.set_ylabel('GB/s (A + x + y)')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right')
    axes[1].set_xlabel('n (n x n matrices)')
    fig.suptitle('Batched gemv, %s (%s)' % (info.get('name', '?'), info.get('arch', '?')))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    log('Plot: ' + path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mfem-dir', required=True,
                    help='MFEM build directory (with config/config.mk) or '
                         'install prefix (with share/mfem/config.mk)')
    ap.add_argument('-o', '--output',
                    help='tuning header to write (default: '
                         'batched_gemv_tuning_<gpu>.hpp in the current directory)')
    ap.add_argument('--work-dir', default='tune_batched_gemv_build',
                    help='directory for the benchmark build and raw results')
    ap.add_argument('--launcher', default='',
                    help='command prefix to run the benchmark on a GPU node, '
                         'e.g. "srun -n1 -G1" or "flux run -n1 -g1"')
    ap.add_argument('--budget-mb', type=float, default=1024,
                    help='MB of A + x + y per test; keep it well above the '
                         'GPU\'s last-level cache (default 1024)')
    ap.add_argument('--trial-ms', type=float, default=20,
                    help='minimum duration of each timing trial (default 20)')
    ap.add_argument('--stride', type=int, default=1,
                    help='measure only every stride-th size above 32 (plus '
                         'the low-order element sizes); default 1 = all')
    ap.add_argument('--quick', action='store_true',
                    help='same as --stride 4 --trial-ms 10')
    ap.add_argument('--fe-weight', type=float, default=4,
                    help='weight of the low-order element sizes %s relative '
                         'to other sizes (default 4)' % FE_SIZES)
    ap.add_argument('--no-vendor', action='store_true',
                    help='do not measure cuBLAS/hipBLAS/MAGMA for comparison')
    ap.add_argument('--skip-build', action='store_true',
                    help='reuse the benchmark already built in --work-dir')
    ap.add_argument('--results',
                    help='do not build or run; tune from this raw results file')
    ap.add_argument('--jobs', type=int, default=0,
                    help='parallel compile jobs (default: number of CPUs)')
    ap.add_argument('--make', default='make', help='GNU make command')
    ap.add_argument('--extra-cxxflags', default='',
                    help='extra compile flags, if the MFEM config.mk misses '
                         'some (e.g. -I<path to RAJA>/include)')
    ap.add_argument('--extra-libs', default='',
                    help='extra link flags and libraries, appended to MFEM_LIBS')
    ap.add_argument('--plot', action='store_true',
                    help='also write a plot (needs matplotlib)')
    ap.add_argument('--apply-cmake', metavar='BUILD_DIR',
                    help='run cmake -DMFEM_BATCHED_GEMV_TUNING=<header> on '
                         'this CMake build directory')
    args = ap.parse_args()
    if args.quick:
        args.stride, args.trial_ms = 4, 10
    args.work_dir = os.path.abspath(args.work_dir)

    mk = make_vars(find_config_mk(args.mfem_dir),
                   ['MFEM_CXX', 'MFEM_FLAGS', 'MFEM_LINK_FLAGS', 'MFEM_LIBS',
                    'MFEM_USE_CUDA', 'MFEM_USE_HIP'], args.make)
    if mk.get('MFEM_USE_HIP') == 'YES':
        backend = 'hip'
    elif mk.get('MFEM_USE_CUDA') == 'YES':
        backend = 'cuda'
    else:
        die('MFEM in %s was not built with CUDA or HIP' % args.mfem_dir)

    if args.results:
        with open(args.results) as f:
            lines = f.readlines()
    else:
        exe = os.path.join(args.work_dir, 'tune_batched_gemv')
        if not (args.skip_build and os.path.isfile(exe)):
            exe = build(args, backend, mk)
        lines = run(args, exe)

    res = Results(lines)
    for fam, cfg, op, n, err in res.bad:
        log('warning: %s %s op %s n %d is wrong (error %.2e), ignored' %
            (fam, cfg, op, n, err))
    tuned = tune(res, args.fe_weight)
    rows, vend = summary_rows(res, tuned)

    if args.output:
        out = os.path.abspath(args.output)
    else:
        gpu = re.sub(r'[^A-Za-z0-9]+', '_', res.info.get('name', 'gpu')).strip('_')
        out = os.path.abspath('batched_gemv_tuning_%s.hpp' % gpu.lower())
    write_header(out, res, tuned, rows, vend, backend, args)

    log('')
    log('GPU: %s (%s)' % (res.info.get('name'), res.info.get('arch')))
    log('%-14s %8s %8s' % ('parameter', 'tuned', 'default'))
    for k in PARAMS:
        log('%-14s %8d %8d%s' % (k, tuned[k], res.default[k],
                                 '' if tuned[k] == res.default[k] else '  *'))
    log('')
    log('GB/s (A + x + y) at the low-order element sizes:')
    log('op    n   tuned  default' + ''.join(' %9s' % c for c in vend))
    for op, n, t, d, b, v in rows:
        if n in FE_SIZES:
            log('%s  %4d  %6.1f  %7.1f' % (op, n, t, d) + ''.join(
                ' %9s' % ('%.1f' % v[c] if v[c] is not None else '-') for c in vend))
    for op in ('N', 'T'):
        r = [x for x in rows if x[0] == op]
        worst = min(r, key=lambda x: x[2] / x[4])
        log('op %s, all %d sizes: tuned/default mean %.3f, min %.3f; worst '
            'tuned vs best kernel %.3f at n = %d' % (
                op, len(r), sum(x[2] / x[3] for x in r) / len(r),
                min(x[2] / x[3] for x in r), worst[2] / worst[4], worst[1]))
    if args.plot:
        plot(os.path.splitext(out)[0] + '.png', rows, vend, res.info)
    log('')
    log('Wrote ' + out)
    if args.apply_cmake:
        cmd = ['cmake', '-DMFEM_BATCHED_GEMV_TUNING=' + out, args.apply_cmake]
        log('Running: ' + ' '.join(cmd))
        if subprocess.run(cmd).returncode != 0:
            die('cmake failed')
        log('Now rebuild (and reinstall) MFEM in ' + args.apply_cmake)
    else:
        log('To use it, reconfigure MFEM and rebuild it:')
        log('  CMake: cmake -DMFEM_BATCHED_GEMV_TUNING=%s <build dir>' % out)
        log('  make:  make config MFEM_BATCHED_GEMV_TUNING=%s <your other '
            'options>' % out)


if __name__ == '__main__':
    main()
