#!/usr/bin/env python3
"""
P4a Conformance Testing — Step 1: Input Sequence Generator
Generates 50 random input sequences x 10 scan cycles for each SAFE-LD benchmark.
"""

import json
import random
import argparse
from pathlib import Path

BENCHMARKS = {
    "motor_interlock": {
        "inputs": ["Forward_Button", "Reverse_Button", "Stop_Button", "Emergency_Stop"],
        "outputs": ["Motor_Forward", "Motor_Reverse", "Alarm"],
        "source": "SAFE-LD original",
        "ld_file": "regression/ld/motor_interlock/guard.ld",
    },
    "traffic_light_safe": {
        "inputs": ["Enable", "Emergency_Vehicle", "Ped_Request"],
        "outputs": ["NS_Green", "NS_Yellow", "NS_Red", "EW_Green", "EW_Yellow", "EW_Red",
                    "Ped_NS_Walk", "Ped_EW_Walk"],
        "source": "Our work",
        "ld_file": "benchmarks/traffic_light/traffic_light_safe.ld",
    },
    "bottle_filling_safe": {
        "inputs": ["Start_Button", "Stop_Button", "Emergency_Stop",
                   "Bottle_Present", "Level_Full", "Conveyor_Ready"],
        "outputs": ["Fill_Valve", "Conveyor_Motor", "Filling_Active",
                    "Alarm_Overfill", "Alarm_No_Bottle"],
        "source": "Our work",
        "ld_file": "benchmarks/bottle_filling/bottle_filling_safe.ld",
    },
    "elevator_safe": {
        "inputs": ["At_Floor1", "At_Floor2", "At_Floor3",
                   "Call_Floor1", "Call_Floor2", "Call_Floor3",
                   "Emergency_Stop", "Door_Closed", "Overload"],
        "outputs": ["Motor_Up", "Motor_Down", "Door_Open", "Alarm_Active", "Motor_Running"],
        "source": "Our work",
        "ld_file": "benchmarks/elevator/elevator_safe.ld",
        "constraints": [
            lambda inp: sum([inp["At_Floor1"], inp["At_Floor2"], inp["At_Floor3"]]) <= 1
        ]
    },
    "water_control": {
        "inputs": ["Pool_Low_Level_Sensor", "Tank_High_Level_Sensor",
                   "Tank_Low_Level_Sensor", "Automatic_Manual_Switch",
                   "Stop_Button", "Start_Button"],
        "outputs": ["Water_Pump"],
        "source": "CONTROLLINO-PLC (MIT, 2024)",
        "ld_file": "benchmarks/water_control/water_control.ld",
        "constraints": [
            lambda inp: not (inp["Tank_High_Level_Sensor"] and inp["Tank_Low_Level_Sensor"])
        ]
    },
    "stairs_light": {
        "inputs": ["stairs_pir_sensor", "control_button_down", "control_button_up"],
        "outputs": ["stairs_light", "lights_buttons_state"],
        "source": "CONTROLLINO-PLC (MIT, 2024)",
        "ld_file": "benchmarks/stairs_light/stairs_light.ld",
    },
    "tank_level_safe": {
        "inputs": ["AUTO_MODE", "HIGH_SWITCH", "LOW_SWITCH"],
        "outputs": ["PUMP", "VALVE"],
        "source": "MathWorks spec (public)",
        "ld_file": "benchmarks/tank_level_control/tank_level_control.ld",
        "constraints": [
            lambda inp: not (inp["HIGH_SWITCH"] and inp["LOW_SWITCH"])
        ]
    },
}

RANDOM_SEED = 42
N_SEQUENCES = 50
N_SCANS     = 10

def generate_input(var_names, constraints=None, max_tries=1000):
    for _ in range(max_tries):
        inp = {v: random.choice([0, 1]) for v in var_names}
        if not constraints or all(c(inp) for c in constraints):
            return inp
    return {v: 0 for v in var_names}

def generate_sequence(benchmark, n_scans):
    constraints = benchmark.get("constraints", [])
    return [generate_input(benchmark["inputs"], constraints) for _ in range(n_scans)]

def generate_all_sequences(benchmark, n_sequences, n_scans):
    return {
        "benchmark_info": {
            "source": benchmark["source"],
            "ld_file": benchmark["ld_file"],
            "n_sequences": n_sequences,
            "n_scans": n_scans,
            "random_seed": RANDOM_SEED,
        },
        "inputs": benchmark["inputs"],
        "outputs": benchmark["outputs"],
        "sequences": [
            {"seq_id": i, "scans": generate_sequence(benchmark, n_scans)}
            for i in range(n_sequences)
        ]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=list(BENCHMARKS.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", default="conformance/inputs")
    parser.add_argument("--sequences", type=int, default=N_SEQUENCES)
    parser.add_argument("--scans", type=int, default=N_SCANS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.benchmark and not args.all:
        parser.print_help()
        return

    targets = list(BENCHMARKS.keys()) if args.all else [args.benchmark]

    print(f"\nP4a Input Generator — {len(targets)} benchmark(s)")
    print(f"  Sequences: {args.sequences}  |  Scans/seq: {args.scans}  |  Seed: {args.seed}\n")

    generated = []
    for name in targets:
        benchmark = BENCHMARKS[name]
        data = generate_all_sequences(benchmark, args.sequences, args.scans)
        out_file = output_dir / f"{name}_inputs.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ {out_file}  ({args.sequences} seqs x {args.scans} scans, {len(benchmark['inputs'])} inputs)")

        if args.preview:
            seq0 = data["sequences"][0]["scans"]
            print("    Preview (first 3 scans):")
            for i, scan in enumerate(seq0[:3]):
                print(f"      scan {i}: {scan}")
        generated.append(str(out_file))

    manifest = {
        "seed": args.seed, "n_sequences": args.sequences,
        "n_scans": args.scans, "benchmarks": targets, "files": generated
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Done. Manifest: {output_dir}/manifest.json")


if __name__ == "__main__":
    main()
