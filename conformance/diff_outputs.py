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
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def scan_diff(plc_scan, esbmc_scan):
    """Return per-variable divergences between one OpenPLC and ESBMC scan."""
    diffs = {}
    for var in sorted(set(plc_scan) | set(esbmc_scan)):
        pv = plc_scan.get(var, "MISSING")
        ev = esbmc_scan.get(var, "MISSING")
        if pv != ev:
            diffs[var] = {"openplc": pv, "esbmc": ev}
    return diffs

def diff_outputs(plc_out, esbmc_out, name):
    divergences = []
    total = 0
    for seq_id, (plc_seq, esbmc_seq) in enumerate(zip(plc_out, esbmc_out)):
        for scan_id, (plc_scan, esbmc_scan) in enumerate(zip(plc_seq, esbmc_seq)):
            total += 1
            diffs = scan_diff(plc_scan, esbmc_scan)
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

def compare_all(openplc_dir, esbmc_dir, benchmarks):
    """Diff every benchmark's OpenPLC and ESBMC outputs and print a summary."""
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
    return results

def print_latex_rows(results):
    """Emit LaTeX table rows for the conformance results."""
    print("\n── LaTeX rows for Table~\\ref{{tab:conformance}} ──")
    for r in results:
        n = r["divergences"]
        cell = r"$\checkmark$" if n == 0 else rf"\textbf{{{n}}}"
        print(f"  {r['benchmark']:<35} & {r['total_comparisons']:>5} & {cell} \\\\")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openplc",  required=True)
    parser.add_argument("--esbmc",    required=True)
    parser.add_argument("--inputs",   required=True)
    parser.add_argument("--report",   default="conformance/conformance_report.json")
    args = parser.parse_args()

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_json(Path(args.inputs) / "manifest.json")

    print("\nP4a Conformance Diff\n")
    results = compare_all(Path(args.openplc), Path(args.esbmc), manifest["benchmarks"])

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Report: {report_path}")

    print_latex_rows(results)

    matched = sum(1 for r in results if r["divergences"] == 0)
    print(f"\n  {matched}/{len(results)} benchmarks fully match.")


if __name__ == "__main__":
    main()
