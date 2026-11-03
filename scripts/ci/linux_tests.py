#!/usr/bin/env python3
"""Plan, run and judge the Linux PR regression split into shards.

See docs/ci/linux-regression.md for the jobs that call each command.
"""

import argparse
import json
import sys

import ci_results
import ci_selection


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=1, sort_keys=True)


def _read_json(path):
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def cmd_plan(args):
    plan = ci_selection.make_plan(args.build_dir)
    _write_json(args.out, plan)
    print(f"plan {plan['digest']}: " + ", ".join(f"{e}={len(n)}"
                                                 for e, n in plan["executors"].items()))
    return 0


def cmd_probe(args):
    probes = ci_results.probe_capabilities(_read_json(args.plan), ci_results.load_testing_tool())
    _write_json(args.out, probes)
    return 0


def cmd_evaluate(args):
    plan = _read_json(args.plan)
    expected = plan["baseline"] if args.executor == "baseline" else plan["executors"][args.executor]
    report = ci_results.run_report(args.executor, expected, args.junit, args.exit_code, plan,
                                   _read_json(args.probes), ci_results.load_testing_tool())
    _write_json(args.out, report)
    for problem in report["problems"]:
        print(f"::error::{args.executor}: {problem}")
    return 1 if report["problems"] else 0


def parser():
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="write the executors' expected tests")
    plan.add_argument("--build-dir", required=True)
    plan.add_argument("--out", required=True)
    plan.set_defaults(run=cmd_plan)

    probe = commands.add_parser("probe", help="record the harness's dynamic capabilities")
    probe.add_argument("--plan", required=True)
    probe.add_argument("--out", required=True)
    probe.set_defaults(run=cmd_probe)

    evaluate = commands.add_parser("evaluate", help="judge one executor's JUnit and exit code")
    evaluate.add_argument("--plan", required=True)
    evaluate.add_argument("--probes", required=True)
    evaluate.add_argument("--executor", required=True)
    evaluate.add_argument("--junit", required=True)
    evaluate.add_argument("--exit-code", type=int, required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.set_defaults(run=cmd_evaluate)
    return top


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return args.run(args)
    except ci_selection.SelectionError as error:
        print(f"::error::{error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
