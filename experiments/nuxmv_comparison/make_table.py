#!/usr/bin/env python3
"""Generate LaTeX comparison table from experiment CSV."""
import csv
import sys


def verdict_cell(v, expected):
    if v == "TIMEOUT":
        return r"\textbf{TIMEOUT}"
    if v == "UNKNOWN":
        return r"\textit{UNKNOWN}"
    color = "green!20" if v == expected else "red!20"
    symbol = r"\checkmark" if v == expected else r"\times"
    return r"\cellcolor{" + color + r"}" + f"{symbol} {v}"


def fmt_time(t, verdict):
    if t in ("N/A", "", None):
        return "N/A"
    try:
        f = float(t)
        if verdict == "TIMEOUT":
            return r"$>$120s"
        return f"{f:.3f}s"
    except ValueError:
        return t


BENCH_LABELS = {
    "tank_level_safe": r"\textsc{TankLevel} (safe)",
    "tank_level_unsafe": r"\textsc{TankLevel} (unsafe)",
    "bottle_filling_safe": r"\textsc{BottleFill} (safe)",
    "bottle_filling_unsafe": r"\textsc{BottleFill} (unsafe)",
    "elevator_safe": r"\textsc{Elevator} (safe)",
    "elevator_unsafe": r"\textsc{Elevator} (unsafe)",
    "traffic_light_safe": r"\textsc{TrafLight} (safe)",
    "traffic_light_unsafe": r"\textsc{TrafLight} (unsafe)",
}


def main(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{ESBMC-PLC+ vs NuXmv BDD and IC3 — performance comparison on "
        r"IEC~61131-3 LD benchmarks. "
        r"ESBMC uses \textit{k}-induction with Z3; NuXmv BDD uses CUDD-based BDD "
        r"reachability; NuXmv IC3 uses the IC3/PDR algorithm. Timeout: 120\,s. "
        r"INT columns show the number of integer-valued timer/counter state variables.}"
    )
    lines.append(r"\label{tab:nuxmv_comparison}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lrrr|rr|rr|rr}")
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{\textbf{Benchmark}} & \multirow{2}{*}{\textbf{INT}} & "
        r"\multirow{2}{*}{\textbf{Rungs}} & \multirow{2}{*}{\textbf{Props}} &"
    )
    lines.append(r"\multicolumn{2}{c|}{\textbf{ESBMC-PLC+ ($k$-ind)}} &")
    lines.append(r"\multicolumn{2}{c|}{\textbf{NuXmv BDD}} &")
    lines.append(r"\multicolumn{2}{c}{\textbf{NuXmv IC3}} \\")
    lines.append(
        r" & & & & \textbf{Result} & \textbf{Time} & \textbf{Result} & \textbf{Time}"
        r" & \textbf{Result} & \textbf{Time} \\"
    )
    lines.append(r"\midrule")

    for row in rows:
        bench = row["benchmark"]
        label = BENCH_LABELS.get(bench, bench)
        exp = row["expected"]
        ev = row["esbmc_verdict"]
        et = row["esbmc_time_s"]
        bv = row["nuxmv_bdd_verdict"]
        bt = row["nuxmv_bdd_time_s"]
        iv = row["nuxmv_ic3_verdict"]
        it_ = row["nuxmv_ic3_time_s"]
        ints = row["num_int_vars"]
        rungs = row["num_rungs"]
        props = row["num_props"]

        e_cell = verdict_cell(ev, exp)
        b_cell = verdict_cell(bv, exp)
        i_cell = verdict_cell(iv, exp)

        e_time = fmt_time(et, ev)
        b_time = fmt_time(bt, bv)
        i_time = fmt_time(it_, iv)

        lines.append(
            f"{label} & {ints} & {rungs} & {props} & "
            f"{e_cell} & {e_time} & "
            f"{b_cell} & {b_time} & "
            f"{i_cell} & {i_time} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
