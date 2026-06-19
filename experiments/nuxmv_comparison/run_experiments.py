#!/usr/bin/env python3
"""
run_experiments.py — ESBMC-PLC+ vs NuXmv BDD/IC3 experiment runner

Produces:
  <outdir>/results.csv        — per-benchmark verdicts and wall-clock times
  <outdir>/results_table.tex  — LaTeX table (Table 3 in the paper)

Usage:
  python3 run_experiments.py [--esbmc PATH] [--nuxmv PATH] [--bench DIR]
                              [--out DIR] [--timeout SEC]
"""
import argparse
import csv
import os
import pathlib
import re
import subprocess
import sys
import time

import yaml

SCRIPT_DIR = pathlib.Path(__file__).parent
TRANSPILER = str(SCRIPT_DIR / "ld_to_smv.py")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument(
    "--esbmc",
    default=os.environ.get("ESBMC", "esbmc"),
    help="Path to ESBMC binary",
)
ap.add_argument(
    "--nuxmv",
    default=os.environ.get("NUXMV", "nuXmv"),
    help="Path to nuXmv binary",
)
ap.add_argument(
    "--bench",
    default=str(SCRIPT_DIR.parent.parent / "benchmarks"),
    help="Root benchmark directory",
)
ap.add_argument(
    "--out",
    default=str(SCRIPT_DIR / "results"),
    help="Output directory",
)
ap.add_argument("--timeout", type=int, default=120, help="Per-tool timeout (seconds)")
args = ap.parse_args()

ESBMC = args.esbmc
NUXMV = args.nuxmv
BENCH_DIR = pathlib.Path(args.bench)
RESULTS_DIR = pathlib.Path(args.out)
TIMEOUT = args.timeout

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SMV_DIR = RESULTS_DIR / "smv"
SMV_DIR.mkdir(exist_ok=True)
LOG_DIR = RESULTS_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

BENCHMARKS = [
    (
        "tank_level_safe",
        BENCH_DIR / "tank_level_control/tank_level_control.ld",
        BENCH_DIR / "tank_level_control/props.yaml",
        "SAFE",
    ),
    (
        "tank_level_unsafe",
        BENCH_DIR / "tank_level_control/tank_level_control_unsafe.ld",
        BENCH_DIR / "tank_level_control/props.yaml",
        "VIOLATION",
    ),
    (
        "bottle_filling_safe",
        BENCH_DIR / "bottle_filling/bottle_filling_safe.ld",
        BENCH_DIR / "bottle_filling/props.yaml",
        "SAFE",
    ),
    (
        "bottle_filling_unsafe",
        BENCH_DIR / "bottle_filling/bottle_filling_unsafe.ld",
        BENCH_DIR / "bottle_filling/props.yaml",
        "VIOLATION",
    ),
    (
        "elevator_safe",
        BENCH_DIR / "elevator/elevator_safe.ld",
        BENCH_DIR / "elevator/props.yaml",
        "SAFE",
    ),
    (
        "elevator_unsafe",
        BENCH_DIR / "elevator/elevator_unsafe.ld",
        BENCH_DIR / "elevator/props_unsafe.yaml",
        "VIOLATION",
    ),
    (
        "traffic_light_safe",
        BENCH_DIR / "traffic_light/traffic_light_safe.ld",
        BENCH_DIR / "traffic_light/props.yaml",
        "SAFE",
    ),
    (
        "traffic_light_unsafe",
        BENCH_DIR / "traffic_light/traffic_light_unsafe.ld",
        BENCH_DIR / "traffic_light/props.yaml",
        "VIOLATION",
    ),
]

FIELDS = [
    "benchmark",
    "expected",
    "esbmc_verdict",
    "esbmc_time_s",
    "nuxmv_bdd_verdict",
    "nuxmv_bdd_time_s",
    "nuxmv_ic3_verdict",
    "nuxmv_ic3_time_s",
    "num_bool_vars",
    "num_int_vars",
    "num_rungs",
    "num_props",
]


# ---------------------------------------------------------------------------
def run_cmd(cmd, log_path=None):
    """Run a command with timeout. Returns (elapsed_s, returncode, output)."""
    start = time.perf_counter()
    try:
        result = subprocess.run(  # nosec B603
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            env=dict(os.environ),
            check=False,
        )
        elapsed = time.perf_counter() - start
        out = result.stdout + result.stderr
        if log_path:
            pathlib.Path(log_path).write_text(out, encoding="utf-8")
        return elapsed, result.returncode, out
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start

        def _dec(b):
            if b is None:
                return ""
            return b.decode("utf-8", errors="replace") if isinstance(b, bytes) else b

        out = _dec(exc.stdout) + _dec(exc.stderr)
        if log_path:
            pathlib.Path(log_path).write_text(out + "\n[TIMEOUT]", encoding="utf-8")
        return elapsed, 124, out


def run_nuxmv(smv_path, mode, log_path):
    """Run nuXmv in BDD or IC3 mode, feeding commands via stdin."""
    if mode == "bdd":
        cmds = (
            f"read_model -i {smv_path}\n"
            "flatten_hierarchy\nencode_variables\nbuild_model\n"
            "check_invar\nquit\n"
        )
    else:
        cmds = (
            f"read_model -i {smv_path}\n"
            "flatten_hierarchy\nencode_variables\nbuild_boolean_model\n"
            "check_invar_ic3\nquit\n"
        )
    start = time.perf_counter()
    try:
        result = subprocess.run(  # nosec B603
            [NUXMV, "-int"],
            input=cmds,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            env=dict(os.environ),
            check=False,
        )
        elapsed = time.perf_counter() - start
        out = result.stdout + result.stderr
        if log_path:
            pathlib.Path(log_path).write_text(out, encoding="utf-8")
        return elapsed, result.returncode, out
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start

        def _dec(b):
            if b is None:
                return ""
            return b.decode("utf-8", errors="replace") if isinstance(b, bytes) else b

        out = _dec(exc.stdout) + _dec(exc.stderr)
        if log_path:
            pathlib.Path(log_path).write_text(out + "\n[TIMEOUT]", encoding="utf-8")
        return elapsed, 124, out


def esbmc_verdict(out):
    if "VERIFICATION SUCCESSFUL" in out:
        return "SAFE"
    if "VERIFICATION FAILED" in out:
        return "VIOLATION"
    return "UNKNOWN"


def nuxmv_verdict(out, rc):
    if rc == 124:
        return "TIMEOUT"
    if re.search(r"is false", out):
        return "VIOLATION"
    if re.search(r"is true", out):
        return "SAFE"
    return "UNKNOWN"


def count_vars(ld_path):
    text = pathlib.Path(ld_path).read_text(encoding="utf-8")
    return (
        len(re.findall(r"<BOOL/>", text)),
        len(re.findall(r"<(?:INT|DINT|UINT|WORD|BYTE)\/>", text)),
        len(re.findall(r"<rung\b", text)),
    )


# ---------------------------------------------------------------------------
csv_path = RESULTS_DIR / "results.csv"
print(f"\nESBMC binary : {ESBMC}")
print(f"nuXmv binary : {NUXMV}")
print(f"Timeout      : {TIMEOUT}s\n")

with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
    writer = csv.DictWriter(csvf, fieldnames=FIELDS)
    writer.writeheader()

    for label, ld_path, props_path, expected in BENCHMARKS:
        ld_path = pathlib.Path(ld_path)
        props_path = pathlib.Path(props_path)
        print(f'\n{"═"*58}')
        print(f"  {label}  (expected: {expected})")

        if not ld_path.exists():
            print(f"  [SKIP] LD not found: {ld_path}")
            continue
        if not props_path.exists():
            print(f"  [SKIP] Props not found: {props_path}")
            continue

        bool_cnt, int_cnt, rung_cnt = count_vars(ld_path)
        with open(props_path, encoding="utf-8") as f:
            prop_cnt = len(yaml.safe_load(f).get("properties", []))
        print(f"  BOOL={bool_cnt} INT={int_cnt} Rungs={rung_cnt} Props={prop_cnt}")

        # Transpile LD -> SMV
        smv_path = str(SMV_DIR / f"{label}.smv")
        _, tp_rc, _ = run_cmd(
            [sys.executable, TRANSPILER, str(ld_path), str(props_path), "--out", smv_path]
        )
        smv_ok = tp_rc == 0 and pathlib.Path(smv_path).exists()
        print(f'  [SMV] {"OK" if smv_ok else "FAILED"}')

        # ESBMC
        esbmc_log = str(LOG_DIR / f"{label}_esbmc.log")
        elapsed_e, rc_e, out_e = run_cmd(
            [
                ESBMC,
                str(ld_path),
                "--ld-props",
                str(props_path),
                "--k-induction",
                "--z3",
                "--no-div-by-zero-check",
            ],
            log_path=esbmc_log,
        )
        v_esbmc = "TIMEOUT" if rc_e == 124 else esbmc_verdict(out_e)
        print(f"  [ESBMC]      {v_esbmc:12s} {elapsed_e:.3f}s")

        # NuXmv BDD
        elapsed_b, v_bdd = 0.0, "N/A"
        if smv_ok:
            bdd_log = str(LOG_DIR / f"{label}_nuxmv_bdd.log")
            elapsed_b, rc_b, out_b = run_nuxmv(smv_path, "bdd", bdd_log)
            v_bdd = nuxmv_verdict(out_b, rc_b)
            print(f"  [NuXmv BDD]  {v_bdd:12s} {elapsed_b:.3f}s")
        else:
            print("  [NuXmv BDD]  N/A (no SMV)")

        # NuXmv IC3
        elapsed_i, v_ic3 = 0.0, "N/A"
        if smv_ok:
            ic3_log = str(LOG_DIR / f"{label}_nuxmv_ic3.log")
            elapsed_i, rc_i, out_i = run_nuxmv(smv_path, "ic3", ic3_log)
            v_ic3 = nuxmv_verdict(out_i, rc_i)
            print(f"  [NuXmv IC3]  {v_ic3:12s} {elapsed_i:.3f}s")
        else:
            print("  [NuXmv IC3]  N/A (no SMV)")

        writer.writerow(
            {
                "benchmark": label,
                "expected": expected,
                "esbmc_verdict": v_esbmc,
                "esbmc_time_s": f"{elapsed_e:.3f}",
                "nuxmv_bdd_verdict": v_bdd,
                "nuxmv_bdd_time_s": f"{elapsed_b:.3f}",
                "nuxmv_ic3_verdict": v_ic3,
                "nuxmv_ic3_time_s": f"{elapsed_i:.3f}",
                "num_bool_vars": bool_cnt,
                "num_int_vars": int_cnt,
                "num_rungs": rung_cnt,
                "num_props": prop_cnt,
            }
        )
        csvf.flush()

print(f'\n{"═"*58}')
print(f"  Results CSV : {csv_path}")

# Generate LaTeX table
make_table = str(SCRIPT_DIR / "make_table.py")
tex_path = RESULTS_DIR / "results_table.tex"
table_result = subprocess.run(  # nosec B603
    [sys.executable, make_table, str(csv_path)],
    capture_output=True,
    text=True,
    check=False,
)
tex_path.write_text(table_result.stdout, encoding="utf-8")
print(f"  LaTeX table : {tex_path}")
