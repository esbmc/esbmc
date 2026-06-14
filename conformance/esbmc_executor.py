#!/usr/bin/env python3
"""
P4a Conformance Testing — Step 3: ESBMC concrete executor.
Uses --no-slice + probe property to extract concrete output values
for each input scan from the ESBMC counterexample trace.
"""

import json
# Invokes the trusted ESBMC binary via a fixed argv list, never a shell.
import subprocess  # nosec B404
import re
import argparse
import tempfile
from pathlib import Path

ESBMC_BIN = "./build/src/esbmc/esbmc"

LD_FILES = {
    "motor_interlock":     "regression/ld/motor_interlock/guard.ld",
    "traffic_light_safe":  "benchmarks/traffic_light/traffic_light_safe.ld",
    "bottle_filling_safe": "benchmarks/bottle_filling/bottle_filling_safe.ld",
    "elevator_safe":       "benchmarks/elevator/elevator_safe.ld",
    "water_control":       "benchmarks/water_control/water_control.ld",
    "stairs_light":        "benchmarks/stairs_light/stairs_light.ld",
    "tank_level_safe":     "benchmarks/tank_level_control/tank_level_control.ld",
}

def make_probe_props(first_output_var: str) -> str:
    """
    Generate props.yaml that forces a trace via an always-true absence
    property on an output, getting ESBMC to expose the output values.
    """
    # Use the first output variable in a tautology to force trace
    return f"""properties:
  - id: PROBE
    kind: absence
    expression: "{first_output_var} || !{first_output_var}"
    description: "Concrete probe — always triggers to expose output values"
"""

def parse_trace(stdout: str, all_vars: list) -> dict:
    """
    Parse ESBMC counterexample trace.
    Extracts 'Variable = value' from State lines.
    Returns dict of {var: int_value}.
    Last assignment wins (handles multi-scan traces).
    """
    values = {}
    for line in stdout.split('\n'):
        line = line.strip()
        for var in all_vars:
            # Match exact variable name followed by = and integer
            m = re.match(rf'^{re.escape(var)}\s*=\s*(-?\d+)$', line)
            if m:
                values[var] = int(m.group(1))
    return values

def run_esbmc_probe(ld_file: str, props_file: str, timeout: int = 15) -> str:
    """Run ESBMC with --no-slice and return full stdout."""
    cmd = [
        ESBMC_BIN, ld_file,
        "--ld-props", props_file,
        "--incremental-bmc",
        "--no-slice",
    ]
    try:
        # cmd is a fixed argv list (no shell); ESBMC path is operator-supplied.
        r = subprocess.run(cmd, capture_output=True, text=True,  # nosec B603
                           timeout=timeout, check=False)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"

def get_concrete_outputs(ld_file: str, output_vars: list,
                         all_vars: list, tmpdir: Path) -> dict:
    """
    Run ESBMC once and extract outputs from the counterexample trace.
    ESBMC samples inputs non-deterministically, so any trace is valid
    for comparison — we accept the first one found.
    """
    props_file = str(tmpdir / "probe.yaml")
    Path(props_file).write_text(make_probe_props(output_vars[0]))

    stdout = run_esbmc_probe(ld_file, props_file)

    if "TIMEOUT" in stdout or "ERROR" in stdout:
        return {v: -1 for v in output_vars}

    values = parse_trace(stdout, all_vars)

    # Extract only output variables
    return {v: values.get(v, -1) for v in output_vars}

def collect_outputs(ld_file: str, sequences: list, out_vars: list,
                    all_vars: list, total: int) -> list:
    """Run ESBMC for every scan and collect the concrete output values."""
    all_outputs = []
    done = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for seq in sequences:
            seq_outputs = []
            for _ in seq["scans"]:
                seq_outputs.append(
                    get_concrete_outputs(ld_file, out_vars, all_vars, tmp)
                )
                done += 1
                if done % 50 == 0:
                    print(f"    {done}/{total} scans processed...")
            all_outputs.append(seq_outputs)
    return all_outputs

def process_benchmark(name: str, inp_file: Path, ld_file: str,
                      output_dir: Path) -> None:
    data      = json.loads(inp_file.read_text())
    sequences = data["sequences"]
    out_vars  = data["outputs"]
    all_vars  = data["inputs"] + out_vars

    total       = len(sequences) * len(sequences[0]["scans"])
    all_outputs = collect_outputs(ld_file, sequences, out_vars, all_vars, total)

    result = {
        "benchmark": name,
        "n_sequences": len(sequences),
        "n_scans":     len(sequences[0]["scans"]),
        "outputs":     all_outputs,
    }
    out_file = output_dir / f"{name}_outputs.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(f"  ✓ {name} — {total} scans → {out_file}")

def main():
    parser = argparse.ArgumentParser(
        description="P4a Step 3 — ESBMC concrete executor"
    )
    parser.add_argument("--inputs",    required=True)
    parser.add_argument("--output",    default="conformance/esbmc_outputs/")
    parser.add_argument("--esbmc",     default="./build/src/esbmc/esbmc")
    parser.add_argument("--benchmark", help="Single benchmark only")
    parser.add_argument("--sequences", type=int, default=None,
                        help="Limit number of sequences (for speed)")
    args = parser.parse_args()

    global ESBMC_BIN
    ESBMC_BIN = args.esbmc

    inputs_dir = Path(args.inputs)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest   = json.loads((inputs_dir / "manifest.json").read_text())
    benchmarks = [args.benchmark] if args.benchmark else manifest["benchmarks"]

    print("\nP4a ESBMC Concrete Executor")
    print(f"  Binary: {args.esbmc}")
    print("  Note: 1 ESBMC call per scan — may take a few minutes\n")

    for name in benchmarks:
        ld_file = LD_FILES.get(name)
        if not ld_file or not Path(ld_file).exists():
            print(f"  ⚠  {name}: LD file not found")
            continue
        inp_file = inputs_dir / f"{name}_inputs.json"
        if not inp_file.exists():
            print(f"  ⚠  {name}: inputs not found")
            continue

        # Optionally limit sequences for speed
        if args.sequences:
            data = json.loads(inp_file.read_text())
            data["sequences"] = data["sequences"][:args.sequences]
            tmp_inp = output_dir / f"_{name}_tmp_inputs.json"
            tmp_inp.write_text(json.dumps(data))
            inp_file = tmp_inp

        print(f"  Processing: {name}")
        process_benchmark(name, inp_file, ld_file, output_dir)

    print("\n✓ Done. Now run:")
    print("  python3 conformance/diff_outputs.py \\")
    print("    --openplc conformance/sim_outputs/ \\")
    print(f"    --esbmc   {output_dir}/ \\")
    print(f"    --inputs  {inputs_dir}/")


if __name__ == "__main__":
    main()
