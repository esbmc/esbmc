#!/usr/bin/env python3
"""
P4a ESBMC Concrete Executor v2.
Fixes input values by embedding them as pre-conditions in the property:
  absence: (in1==v1 && in2==v2 && ...) && (out || !out)
This forces Z3 to find a trace with exactly these input values,
revealing what the LD program computes for those inputs.
"""

import json, subprocess, re, argparse, tempfile
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

def make_constrained_props(inputs: dict, probe_var: str) -> str:
    """
    Property: absence of (all inputs match) AND (probe tautology).
    Forces ESBMC to find a trace where inputs = concrete values,
    revealing the output values computed by the LD program.
    """
    # Build input constraint: each var == its value
    constraints = " && ".join(
        f"({var} == {val})" for var, val in inputs.items()
    )
    # Combined: inputs match AND tautology on output → always triggers
    expr = f"({constraints}) && ({probe_var} || !{probe_var})"
    return f"""properties:
  - id: CONCRETE_PROBE
    kind: absence
    expression: "{expr}"
    description: "Concrete input probe"
"""

def parse_trace(stdout: str, all_vars: list) -> dict:
    values = {}
    for line in stdout.split('\n'):
        line = line.strip()
        for var in all_vars:
            m = re.match(rf'^{re.escape(var)}\s*=\s*(-?\d+)$', line)
            if m:
                values[var] = int(m.group(1))
    return values

def run_esbmc(ld_file: str, props_file: str, timeout: int = 15) -> str:
    cmd = [ESBMC_BIN, ld_file, "--ld-props", props_file,
           "--incremental-bmc", "--no-slice"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"

def get_outputs(ld_file, scan_inputs, output_vars, all_vars, tmpdir):
    probe_var  = output_vars[0]
    props_yaml = make_constrained_props(scan_inputs, probe_var)
    props_file = str(tmpdir / "props.yaml")
    Path(props_file).write_text(props_yaml)

    stdout = run_esbmc(ld_file, props_file)

    if "TIMEOUT" in stdout or "ERROR:" in stdout:
        return {v: -1 for v in output_vars}

    # If VERIFICATION UNKNOWN or no violation, inputs may be unsatisfiable
    if "VERIFICATION FAILED" not in stdout:
        return {v: -2 for v in output_vars}  # -2 = no trace found

    values = parse_trace(stdout, all_vars)
    return {v: values.get(v, -1) for v in output_vars}

def process_benchmark(name, inp_file, ld_file, output_dir, limit_seqs=None):
    data      = json.loads(inp_file.read_text())
    sequences = data["sequences"]
    if limit_seqs:
        sequences = sequences[:limit_seqs]
    inp_vars  = data["inputs"]
    out_vars  = data["outputs"]
    all_vars  = inp_vars + out_vars

    total = len(sequences) * len(sequences[0]["scans"])
    done  = 0
    all_outputs = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for seq in sequences:
            seq_outputs = []
            for scan_inputs in seq["scans"]:
                outputs = get_outputs(ld_file, scan_inputs, out_vars, all_vars, tmp)
                seq_outputs.append(outputs)
                done += 1
                if done % 20 == 0:
                    pct = done * 100 // total
                    print(f"    [{pct:3d}%] {done}/{total} scans")
            all_outputs.append(seq_outputs)

    result = {"benchmark": name, "n_sequences": len(sequences),
              "n_scans": len(sequences[0]["scans"]), "outputs": all_outputs}
    out_file = output_dir / f"{name}_outputs.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(f"  ✓ {name} — {total} scans done")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs",    required=True)
    parser.add_argument("--output",    default="conformance/esbmc_outputs_v2/")
    parser.add_argument("--esbmc",     default="./build/src/esbmc/esbmc")
    parser.add_argument("--benchmark", help="Single benchmark only")
    parser.add_argument("--sequences", type=int, default=None)
    args = parser.parse_args()

    global ESBMC_BIN
    ESBMC_BIN = args.esbmc

    inputs_dir = Path(args.inputs)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest   = json.loads((inputs_dir / "manifest.json").read_text())
    benchmarks = [args.benchmark] if args.benchmark else manifest["benchmarks"]

    print(f"\nP4a ESBMC Concrete Executor v2 (constrained inputs)\n")

    for name in benchmarks:
        ld_file = LD_FILES.get(name)
        if not ld_file or not Path(ld_file).exists():
            print(f"  ⚠  {name}: LD file not found"); continue
        inp_file = inputs_dir / f"{name}_inputs.json"
        if not inp_file.exists():
            print(f"  ⚠  {name}: inputs not found"); continue
        print(f"  → {name}")
        process_benchmark(name, inp_file, ld_file, output_dir, args.sequences)

    print(f"\n✓ Done. Compare with:")
    print(f"  python3 conformance/diff_outputs.py \\")
    print(f"    --openplc conformance/sim_outputs/ \\")
    print(f"    --esbmc   {output_dir}/ \\")
    print(f"    --inputs  {inputs_dir}/")

if __name__ == "__main__":
    main()
