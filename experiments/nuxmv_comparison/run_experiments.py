#!/usr/bin/env python3
"""
run_experiments.py — ESBMC-PLC+ vs NuXmv BDD/IC3 experiment runner
Produces results/results.csv and results/results_table.tex
"""
import os, sys, subprocess, time, signal, csv, pathlib, re, textwrap

SCRIPT_DIR  = pathlib.Path(__file__).parent
BENCH_DIR   = pathlib.Path('/Users/pierredantas/esbmc/benchmarks')
ESBMC       = '/Users/pierredantas/esbmc/build/src/esbmc/esbmc'
NUXMV       = '/tmp/nuXmv-2.2.0-macos64/usr/local/bin/nuXmv'
NUXMV_LIBS  = '/tmp/nuXmv-2.2.0-macos64/usr/local/lib'
TRANSPILER  = str(SCRIPT_DIR / 'ld_to_smv.py')
RESULTS_DIR = SCRIPT_DIR / 'results'
SMV_DIR     = RESULTS_DIR / 'smv'
TIMEOUT     = 120   # seconds

RESULTS_DIR.mkdir(exist_ok=True)
SMV_DIR.mkdir(exist_ok=True)

BENCHMARKS = [
    # (label,                 ld_file,                                         props_file,                                 expected)
    ('tank_level_safe',
     BENCH_DIR/'tank_level_control/tank_level_control.ld',
     BENCH_DIR/'tank_level_control/props.yaml', 'SAFE'),
    ('tank_level_unsafe',
     BENCH_DIR/'tank_level_control/tank_level_control_unsafe.ld',
     BENCH_DIR/'tank_level_control/props.yaml', 'VIOLATION'),
    ('bottle_filling_safe',
     BENCH_DIR/'bottle_filling/bottle_filling_safe.ld',
     BENCH_DIR/'bottle_filling/props.yaml', 'SAFE'),
    ('bottle_filling_unsafe',
     BENCH_DIR/'bottle_filling/bottle_filling_unsafe.ld',
     BENCH_DIR/'bottle_filling/props.yaml', 'VIOLATION'),
    ('elevator_safe',
     BENCH_DIR/'elevator/elevator_safe.ld',
     BENCH_DIR/'elevator/props.yaml', 'SAFE'),
    ('elevator_unsafe',
     BENCH_DIR/'elevator/elevator_unsafe.ld',
     BENCH_DIR/'elevator/props_unsafe.yaml', 'VIOLATION'),
    ('traffic_light_safe',
     BENCH_DIR/'traffic_light/traffic_light_safe.ld',
     BENCH_DIR/'traffic_light/props.yaml', 'SAFE'),
    ('traffic_light_unsafe',
     BENCH_DIR/'traffic_light/traffic_light_unsafe.ld',
     BENCH_DIR/'traffic_light/props.yaml', 'VIOLATION'),
]


# ---------------------------------------------------------------------------
def run_cmd(args, env=None, log_path=None):
    """Run a command with timeout. Returns (elapsed_s, returncode, stdout)."""
    env_ = dict(os.environ)
    if env:
        env_.update(env)
    start = time.perf_counter()
    try:
        result = subprocess.run(
            args, capture_output=True, text=True,
            timeout=TIMEOUT, env=env_
        )
        elapsed = time.perf_counter() - start
        out = result.stdout + result.stderr
        if log_path:
            pathlib.Path(log_path).write_text(out)
        return elapsed, result.returncode, out
    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - start
        def _dec(b):
            if b is None: return ''
            return b.decode('utf-8', errors='replace') if isinstance(b, bytes) else b
        out = _dec(e.stdout) + _dec(e.stderr)
        if log_path:
            pathlib.Path(log_path).write_text(out + '\n[TIMEOUT]')
        return elapsed, 124, out


def run_nuxmv(smv_path, mode='bdd', log_path=None):
    """Run NuXmv with BDD or IC3 mode on smv_path."""
    if mode == 'bdd':
        cmds = textwrap.dedent(f"""\
            read_model -i {smv_path}
            flatten_hierarchy
            encode_variables
            build_model
            check_invar
            quit
        """)
    else:
        cmds = textwrap.dedent(f"""\
            read_model -i {smv_path}
            flatten_hierarchy
            encode_variables
            build_boolean_model
            check_invar_ic3
            quit
        """)
    nuxmv_env = {'DYLD_LIBRARY_PATH': NUXMV_LIBS}
    return run_cmd(
        [NUXMV, '-int'],
        env=nuxmv_env,
        log_path=log_path,
    )
    # Pass commands via stdin — use Popen for stdin pipe
    env_ = dict(os.environ)
    env_.update(nuxmv_env)
    start = time.perf_counter()
    try:
        result = subprocess.run(
            [NUXMV, '-int'], input=cmds,
            capture_output=True, text=True,
            timeout=TIMEOUT, env=env_
        )
        elapsed = time.perf_counter() - start
        out = result.stdout + result.stderr
        if log_path:
            pathlib.Path(log_path).write_text(out)
        return elapsed, result.returncode, out
    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - start
        def _dec(b):
            if b is None: return ''
            return b.decode('utf-8', errors='replace') if isinstance(b, bytes) else b
        out = _dec(e.stdout) + _dec(e.stderr)
        if log_path:
            pathlib.Path(log_path).write_text(out + '\n[TIMEOUT]')
        return elapsed, 124, out


def _run_nuxmv_with_stdin(smv_path, mode, log_path):
    if mode == 'bdd':
        cmds = (f'read_model -i {smv_path}\n'
                'flatten_hierarchy\nencode_variables\nbuild_model\n'
                'check_invar\nquit\n')
    else:
        cmds = (f'read_model -i {smv_path}\n'
                'flatten_hierarchy\nencode_variables\nbuild_boolean_model\n'
                'check_invar_ic3\nquit\n')
    env_ = dict(os.environ)
    env_['DYLD_LIBRARY_PATH'] = NUXMV_LIBS
    start = time.perf_counter()
    try:
        result = subprocess.run(
            [NUXMV, '-int'], input=cmds,
            capture_output=True, text=True,
            timeout=TIMEOUT, env=env_
        )
        elapsed = time.perf_counter() - start
        out = result.stdout + result.stderr
        if log_path:
            pathlib.Path(log_path).write_text(out)
        return elapsed, result.returncode, out
    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - start
        def _dec(b):
            if b is None: return ''
            return b.decode('utf-8', errors='replace') if isinstance(b, bytes) else b
        out = _dec(e.stdout) + _dec(e.stderr)
        if log_path:
            pathlib.Path(log_path).write_text(out + '\n[TIMEOUT]')
        return elapsed, 124, out


# ---------------------------------------------------------------------------
def esbmc_verdict(out):
    if 'VERIFICATION SUCCESSFUL' in out:
        return 'SAFE'
    if 'VERIFICATION FAILED' in out:
        return 'VIOLATION'
    return 'UNKNOWN'


def nuxmv_verdict(out, rc):
    if rc == 124:
        return 'TIMEOUT'
    # INVARSPEC: "invariant ... is true" or "is false"
    if re.search(r'is false', out):
        return 'VIOLATION'
    if re.search(r'is true', out):
        return 'SAFE'
    return 'UNKNOWN'


def count_vars(ld_path):
    """Count BOOL and INT vars and rungs from XML."""
    text = pathlib.Path(ld_path).read_text()
    bool_cnt = len(re.findall(r'<BOOL/>', text))
    int_cnt  = len(re.findall(r'<(?:INT|DINT|UINT|WORD|BYTE)\/>', text))
    rung_cnt = len(re.findall(r'<rung\b', text))
    return bool_cnt, int_cnt, rung_cnt


# ---------------------------------------------------------------------------
FIELDS = [
    'benchmark', 'expected',
    'esbmc_verdict', 'esbmc_time_s',
    'nuxmv_bdd_verdict', 'nuxmv_bdd_time_s',
    'nuxmv_ic3_verdict', 'nuxmv_ic3_time_s',
    'num_bool_vars', 'num_int_vars', 'num_rungs', 'num_props',
]

csv_path = RESULTS_DIR / 'results.csv'

with open(csv_path, 'w', newline='') as csvf:
    writer = csv.DictWriter(csvf, fieldnames=FIELDS)
    writer.writeheader()

    for (label, ld_path, props_path, expected) in BENCHMARKS:
        ld_path    = pathlib.Path(ld_path)
        props_path = pathlib.Path(props_path)
        print(f'\n{"═"*60}')
        print(f'  {label}  (expected: {expected})')

        if not ld_path.exists():
            print(f'  [SKIP] LD not found: {ld_path}')
            continue
        if not props_path.exists():
            print(f'  [SKIP] Props not found: {props_path}')
            continue

        bool_cnt, int_cnt, rung_cnt = count_vars(ld_path)
        import yaml
        with open(props_path) as f:
            prop_cnt = len(yaml.safe_load(f).get('properties', []))

        print(f'  BOOL={bool_cnt} INT={int_cnt} Rungs={rung_cnt} Props={prop_cnt}')

        # ── Transpile LD → SMV ────────────────────────────────────────────
        smv_path = str(SMV_DIR / f'{label}.smv')
        tp_elapsed, tp_rc, tp_out = run_cmd(
            ['python3', TRANSPILER, str(ld_path), str(props_path), '--out', smv_path]
        )
        smv_ok = tp_rc == 0 and pathlib.Path(smv_path).exists()
        print(f'  [SMV] {"OK" if smv_ok else "FAIL"} ({tp_elapsed:.2f}s)')

        # ── ESBMC-PLC ─────────────────────────────────────────────────────
        esbmc_log = str(RESULTS_DIR / f'{label}_esbmc.log')
        elapsed_e, rc_e, out_e = run_cmd(
            [ESBMC, str(ld_path),
             '--ld-props', str(props_path),
             '--k-induction', '--z3', '--no-div-by-zero-check'],
            log_path=esbmc_log
        )
        v_esbmc = esbmc_verdict(out_e)
        if rc_e == 124:
            v_esbmc = 'TIMEOUT'
        print(f'  [ESBMC]      {v_esbmc:10s}  {elapsed_e:.3f}s')

        # ── NuXmv BDD ─────────────────────────────────────────────────────
        bdd_log = str(RESULTS_DIR / f'{label}_nuxmv_bdd.log')
        if smv_ok:
            elapsed_b, rc_b, out_b = _run_nuxmv_with_stdin(smv_path, 'bdd', bdd_log)
            v_bdd = nuxmv_verdict(out_b, rc_b)
            print(f'  [NuXmv BDD]  {v_bdd:10s}  {elapsed_b:.3f}s')
        else:
            elapsed_b, v_bdd = 0.0, 'N/A'
            print(f'  [NuXmv BDD]  N/A (no SMV)')

        # ── NuXmv IC3 ─────────────────────────────────────────────────────
        ic3_log = str(RESULTS_DIR / f'{label}_nuxmv_ic3.log')
        if smv_ok:
            elapsed_i, rc_i, out_i = _run_nuxmv_with_stdin(smv_path, 'ic3', ic3_log)
            v_ic3 = nuxmv_verdict(out_i, rc_i)
            print(f'  [NuXmv IC3]  {v_ic3:10s}  {elapsed_i:.3f}s')
        else:
            elapsed_i, v_ic3 = 0.0, 'N/A'
            print(f'  [NuXmv IC3]  N/A (no SMV)')

        writer.writerow({
            'benchmark':         label,
            'expected':          expected,
            'esbmc_verdict':     v_esbmc,
            'esbmc_time_s':      f'{elapsed_e:.3f}',
            'nuxmv_bdd_verdict': v_bdd,
            'nuxmv_bdd_time_s':  f'{elapsed_b:.3f}' if isinstance(elapsed_b, float) else elapsed_b,
            'nuxmv_ic3_verdict': v_ic3,
            'nuxmv_ic3_time_s':  f'{elapsed_i:.3f}' if isinstance(elapsed_i, float) else elapsed_i,
            'num_bool_vars':     bool_cnt,
            'num_int_vars':      int_cnt,
            'num_rungs':         rung_cnt,
            'num_props':         prop_cnt,
        })
        csvf.flush()

print(f'\n{"═"*60}')
print(f'  Results written to: {csv_path}')

# Generate LaTeX table
import subprocess
make_table = str(SCRIPT_DIR / 'make_table.py')
tex_path = str(RESULTS_DIR / 'results_table.tex')
result = subprocess.run(
    ['python3', make_table, str(csv_path)],
    capture_output=True, text=True
)
pathlib.Path(tex_path).write_text(result.stdout)
print(f'  LaTeX table: {tex_path}')
