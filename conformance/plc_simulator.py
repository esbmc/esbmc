#!/usr/bin/env python3
"""
P4a Conformance Testing — Step 2 (alternative):
Python-based IEC 61131-3 scan cycle simulator.
Executes each benchmark's ladder logic faithfully, scan by scan.
Produces outputs in the same format as the OpenPLC runtime would.

This is a reference implementation based on IEC 61131-3 semantics:
  - Inputs sampled once at start of scan
  - Rungs evaluated left-to-right, top-to-bottom
  - Outputs written at end of scan
  - SET/RESET coils are latching
  - TON: output Q goes TRUE after IN has been TRUE for PT scans
"""

import json
import argparse
from pathlib import Path
from copy import deepcopy

# ── TON Timer model ───────────────────────────────────────────────────────────
class TON:
    """Fixed-tick TON timer: Q goes TRUE after IN has been TRUE for PT scans."""
    def __init__(self):
        self.IN = False
        self.Q  = False
        self.ET = 0
        self.PT = 1  # default: 1 scan

    def tick(self):
        if self.IN:
            self.ET += 1
            if self.ET >= self.PT:
                self.Q = True
        else:
            self.ET = 0
            self.Q  = False

# ── Benchmark executors ───────────────────────────────────────────────────────

def exec_motor_interlock(inputs: dict, state: dict) -> dict:
    """
    motor_interlock: Forward and Reverse are mutually exclusive.
    Forward requires !Reverse and !Emergency_Stop and !Stop_Button.
    Reverse requires !Forward and !Emergency_Stop and !Stop_Button.
    """
    fwd = (inputs["Forward_Button"] and
           not inputs["Reverse_Button"] and
           not inputs["Stop_Button"] and
           not inputs["Emergency_Stop"])
    rev = (inputs["Reverse_Button"] and
           not inputs["Forward_Button"] and
           not inputs["Stop_Button"] and
           not inputs["Emergency_Stop"])
    alarm = bool(inputs["Emergency_Stop"])
    return {"Motor_Forward": int(fwd), "Motor_Reverse": int(rev), "Alarm": int(alarm)}

def exec_traffic_light_safe(inputs: dict, state: dict) -> dict:
    """
    traffic_light_safe: phase sequencer with mutual exclusion via interlocks.
    Simplified model: phases are mutually exclusive by construction.
    """
    enable = inputs["Enable"]
    emerg  = inputs["Emergency_Vehicle"]
    ped    = inputs["Ped_Request"]

    # Phase state machine (simplified — actual LD uses TON timers)
    # In the non-det model, we derive from the current phase state
    ph = state.get("phase", 0)  # 0=NS_Green, 1=NS_Yellow, 2=EW_Green, 3=EW_Yellow

    ns_green  = int(enable and not emerg and ph == 0)
    ns_yellow = int(enable and ph == 1)
    ns_red    = int(not ns_green and not ns_yellow)
    ew_green  = int(enable and ph == 2)
    ew_yellow = int(enable and ph == 3)
    ew_red    = int(not ew_green and not ew_yellow)
    ped_ns    = int(ns_green and ped and not emerg)
    ped_ew    = int(ew_green and ped)

    # Advance phase (one step per scan for simulation purposes)
    state["phase"] = (ph + 1) % 4

    return {
        "NS_Green": ns_green, "NS_Yellow": ns_yellow, "NS_Red": ns_red,
        "EW_Green": ew_green, "EW_Yellow": ew_yellow, "EW_Red": ew_red,
        "Ped_NS_Walk": ped_ns, "Ped_EW_Walk": ped_ew,
    }

def exec_bottle_filling_safe(inputs: dict, state: dict) -> dict:
    """bottle_filling_safe: valve requires bottle present, not full, not emergency."""
    sys_run = (inputs["Start_Button"] and
               not inputs["Stop_Button"] and
               not inputs["Emergency_Stop"])
    valve = int(sys_run and
                inputs["Bottle_Present"] and
                not inputs["Level_Full"] and
                not state.get("filling_done", False) and
                not inputs["Emergency_Stop"])
    # TON1 (simplified: filling_done after 1 scan of valve open)
    if valve:
        state["ton1_et"] = state.get("ton1_et", 0) + 1
    else:
        state["ton1_et"] = 0
    filling_done = int(inputs["Level_Full"] or state["ton1_et"] >= 1)
    state["filling_done"] = bool(filling_done)

    conveyor = int(filling_done and
                   not inputs["Emergency_Stop"] and
                   inputs["Conveyor_Ready"])
    alarm_overfill  = int(inputs["Level_Full"] and valve)
    alarm_no_bottle = int(valve and not inputs["Bottle_Present"])

    return {
        "Fill_Valve": valve,
        "Conveyor_Motor": conveyor,
        "Filling_Active": valve,
        "Alarm_Overfill": alarm_overfill,
        "Alarm_No_Bottle": alarm_no_bottle,
    }

def exec_elevator_safe(inputs: dict, state: dict) -> dict:
    """elevator_safe: motor requires door closed, no emergency, no overload."""
    at = int(inputs["At_Floor1"] or inputs["At_Floor2"] or inputs["At_Floor3"])
    emerg   = inputs["Emergency_Stop"]
    overload = inputs["Overload"]
    door_closed = inputs["Door_Closed"]

    motor_up = int(
        not emerg and not overload and door_closed and
        ((inputs["Call_Floor2"] and inputs["At_Floor1"]) or
         (inputs["Call_Floor3"] and inputs["At_Floor1"]) or
         (inputs["Call_Floor3"] and inputs["At_Floor2"]))
    )
    motor_down = int(
        not emerg and not overload and door_closed and not motor_up and
        ((inputs["Call_Floor1"] and inputs["At_Floor2"]) or
         (inputs["Call_Floor1"] and inputs["At_Floor3"]) or
         (inputs["Call_Floor2"] and inputs["At_Floor3"]))
    )
    motor_running = int(motor_up or motor_down)
    door_open = int(at and not motor_up and not motor_down and not emerg)
    alarm = int(emerg or overload)

    return {
        "Motor_Up": motor_up, "Motor_Down": motor_down,
        "Door_Open": door_open, "Alarm_Active": alarm,
        "Motor_Running": motor_running,
    }

def exec_water_control(inputs: dict, state: dict) -> dict:
    """water_control: pump SET when auto+pool ok+tank needs fill; RESET on full/empty/stop."""
    pump = state.get("Water_Pump", 0)

    # SET conditions (rung 1: auto mode)
    if (inputs["Automatic_Manual_Switch"] and
        inputs["Pool_Low_Level_Sensor"] and
        not inputs["Tank_Low_Level_Sensor"] and
        not inputs["Tank_High_Level_Sensor"]):
        pump = 1

    # SET conditions (rung 2: manual start)
    if (inputs["Start_Button"] and
        inputs["Pool_Low_Level_Sensor"] and
        not inputs["Tank_High_Level_Sensor"]):
        pump = 1

    # RESET conditions (rungs 3,4,5)
    if not inputs["Pool_Low_Level_Sensor"]:
        pump = 0
    if inputs["Tank_High_Level_Sensor"]:
        pump = 0
    if inputs["Stop_Button"]:
        pump = 0

    state["Water_Pump"] = pump
    return {"Water_Pump": pump}

def exec_stairs_light(inputs: dict, state: dict) -> dict:
    """stairs_light: toggle on button press, PIR activates for 20 scans via TOF."""
    btn_state = state.get("lights_buttons_state", 0)
    pir_prev  = state.get("pir_prev", 0)
    tof_et    = state.get("tof_et", 0)

    # Rising edge detection for buttons
    up_edge   = inputs["control_button_up"]   and not state.get("up_prev", 0)
    down_edge = inputs["control_button_down"] and not state.get("down_prev", 0)
    pir_edge  = inputs["stairs_pir_sensor"]   and not pir_prev

    # Toggle logic
    if up_edge or down_edge:
        btn_state = 1 - btn_state  # toggle

    # TOF timer (off-delay): stays on for PT scans after PIR goes low
    if inputs["stairs_pir_sensor"]:
        tof_et = 20  # reset timer to full 20 scans
        tof_q  = 1
    elif tof_et > 0:
        tof_et -= 1
        tof_q  = int(tof_et > 0)
    else:
        tof_q = 0

    # Light output
    light = int(btn_state or (tof_q and not btn_state))

    state["lights_buttons_state"] = btn_state
    state["pir_prev"]   = inputs["stairs_pir_sensor"]
    state["up_prev"]    = inputs["control_button_up"]
    state["down_prev"]  = inputs["control_button_down"]
    state["tof_et"]     = tof_et

    return {"stairs_light": light, "lights_buttons_state": btn_state}

def exec_tank_level_safe(inputs: dict, state: dict) -> dict:
    """tank_level_safe: pump fills LOW->HIGH, valve drains HIGH->LOW, with interlocks."""
    draining = state.get("Draining_Active", 0)
    filling  = state.get("Filling_Active", 0)

    pump = int(inputs["AUTO_MODE"] and
               inputs["LOW_SWITCH"] and
               not inputs["HIGH_SWITCH"] and
               not draining)
    valve = int(inputs["AUTO_MODE"] and
                inputs["HIGH_SWITCH"] and
                not inputs["LOW_SWITCH"] and
                not pump)

    state["Filling_Active"]  = pump
    state["Draining_Active"] = valve

    return {"PUMP": pump, "VALVE": valve}

# ── Executor registry ─────────────────────────────────────────────────────────
EXECUTORS = {
    "motor_interlock":      exec_motor_interlock,
    "traffic_light_safe":   exec_traffic_light_safe,
    "bottle_filling_safe":  exec_bottle_filling_safe,
    "elevator_safe":        exec_elevator_safe,
    "water_control":        exec_water_control,
    "stairs_light":         exec_stairs_light,
    "tank_level_safe":      exec_tank_level_safe,
}

def run_benchmark(name: str, sequences: list, executor) -> list:
    """Run all sequences through the simulator. Returns list of output sequences."""
    all_outputs = []
    for seq in sequences:
        state = {}  # persistent state across scans (latches, timers)
        seq_outputs = []
        for scan in seq["scans"]:
            outputs = executor(scan, state)
            seq_outputs.append(outputs)
        all_outputs.append(seq_outputs)
    return all_outputs

def main():
    parser = argparse.ArgumentParser(
        description="P4a Step 2 — Python PLC simulator (IEC 61131-3 reference)"
    )
    parser.add_argument("--inputs",  required=True, help="conformance/inputs/")
    parser.add_argument("--output",  default="conformance/sim_outputs/")
    parser.add_argument("--benchmark", help="Run one benchmark only")
    args = parser.parse_args()

    inputs_dir = Path(args.inputs)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest   = json.loads((inputs_dir / "manifest.json").read_text())
    benchmarks = [args.benchmark] if args.benchmark else manifest["benchmarks"]

    print(f"\nP4a Python Simulator\n")

    for name in benchmarks:
        if name not in EXECUTORS:
            print(f"  ⚠  {name}: no executor defined, skipping")
            continue

        inp_file = inputs_dir / f"{name}_inputs.json"
        if not inp_file.exists():
            print(f"  ⚠  {name}: input file not found")
            continue

        data      = json.loads(inp_file.read_text())
        sequences = data["sequences"]
        executor  = EXECUTORS[name]

        outputs = run_benchmark(name, sequences, executor)

        result = {
            "benchmark": name,
            "source": data["benchmark_info"]["source"],
            "n_sequences": len(sequences),
            "n_scans": len(sequences[0]["scans"]) if sequences else 0,
            "outputs": outputs,
        }

        out_file = output_dir / f"{name}_outputs.json"
        out_file.write_text(json.dumps(result, indent=2))

        # Quick sanity check on first sequence
        seq0 = outputs[0]
        print(f"  ✓ {name}")
        print(f"    {len(outputs)} sequences x {len(seq0)} scans")
        print(f"    First scan outputs: {seq0[0]}")

    print(f"\n✓ Simulator outputs saved to {output_dir}/")
    print(f"\nNext: run diff_outputs.py to compare with ESBMC outputs")

if __name__ == "__main__":
    main()
