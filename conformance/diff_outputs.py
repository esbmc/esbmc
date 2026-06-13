#!/usr/bin/env python3
"""
P4a Conformance Testing — Step 4: Cycle-by-cycle diff
Compares OpenPLC v3 output vs ESBMC concrete execution output.
Fills the TOINSERT cells for Table~\ref{tab:conformance}.
"""

import json
import argparse
from pathlib import Path

def load_json(path):
    with open(path) as f:
        return json.load(f)

def diff_outputs(plc_out, esbmc_out, name):
    divergences = []
    total = 0
    for seq_id, (plc_seq, esbmc_seq) in enumerate(zip(plc_out, esbmc_out)):
        for scan_id, (plc_scan, esbmc_scan) in enumerate(zip(plc_seq, esbmc_seq)):
            total += 1
            diffs = {}
            for var in sorted(set(plc_scan) | set(esbmc_scan)):
                pv = plc_scan.get(var, "MISSING")
                ev = esbmc_scan.get(var, "MISSING")
                if pv != ev:
                    diffs[var] = {"openplc": pv, "esbmc": ev}
            if diffs:
                divergences.append({"seq_id": seq_id, "scan_id": scan_id, "diffs": diffs})
    n = len(divergences)
    return {
        "benchmark": name, "total_comparisons": total,
        "divergences": n,
        "divergence_rate": f"{n/max(total,1)*100:.1f}%",
        "status": "MATCH" if n == 0 else "DIVERGENCE",
        "details": divergences[:10]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openplc",  required=True)
    parser.add_argument("--esbmc",    required=True)
    parser.add_argument("--inputs",   required=True)
    parser.add_argument("--report",   default="conformance/conformance_report.json")
    args = parser.parse_args()

    openplc_dir = Path(args.openplc)
    esbmc_dir   = Path(args.esbmc)
    inputs_dir  = Path(args.inputs)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    manifest   = load_json(inputs_dir / "manifest.json")
    benchmarks = manifest["benchmarks"]

    print(f"\nP4a Conformance Diff\n")
    results = []
    for name in benchmarks:
        plc_f   = openplc_dir / f"{name}_outputs.json"
        esbmc_f = esbmc_dir   / f"{name}_outputs.json"
        if not plc_f.exists():
            print(f"  ⚠  {name}: missing OpenPLC output"); continue
        if not esbmc_f.exists():
            print(f"  ⚠  {name}: missing ESBMC output"); continue
        result = diff_outputs(load_json(plc_f)["outputs"],
                              load_json(esbmc_f)["outputs"], name)
        results.append(result)
        icon = "✓" if result["divergences"] == 0 else "✗"
        print(f"  {icon} {name:<35} {result['divergences']:>4} div / "
              f"{result['total_comparisons']} total  ({result['divergence_rate']})")

    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Report: {report_path}")

    print("\n── LaTeX rows for Table~\\ref{{tab:conformance}} ──")
    for r in results:
        n = r["divergences"]
        cell = r"$\checkmark$" if n == 0 else rf"\textbf{{{n}}}"
        print(f"  {r['benchmark']:<35} & {r['total_comparisons']:>5} & {cell} \\\\")

    matched = sum(1 for r in results if r["divergences"] == 0)
    print(f"\n  {matched}/{len(results)} benchmarks fully match.")

if __name__ == "__main__":
    main()
