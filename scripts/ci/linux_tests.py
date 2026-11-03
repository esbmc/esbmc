#!/usr/bin/env python3
"""Plan, run and judge the Linux PR regression split into shards.

See docs/ci/linux-regression.md for the jobs that call each command.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import ci_bundle
import ci_gate
import ci_metrics
import ci_results
import ci_selection


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=1, sort_keys=True)


def _read_json(path):
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def _expected(plan, executor):
    return plan["baseline"] if executor == "baseline" else plan["executors"][executor]


def _identity(bundle_sha256):
    return {
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "bundle_sha256": bundle_sha256,
    }


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


def _report(args, plan, junit, exit_code):
    report = ci_results.run_report(args.executor, _expected(plan, args.executor), junit, exit_code,
                                   plan, _read_json(args.probes), ci_results.load_testing_tool())
    for problem in report["problems"]:
        print(f"::error::{args.executor}: {problem}")
    return report


def cmd_evaluate(args):
    report = _report(args, _read_json(args.plan), args.junit, args.exit_code)
    _write_json(args.out, report)
    return 1 if report["problems"] else 0


def cmd_run(args):
    """Run one executor's selection under CTest and judge it."""
    plan = _read_json(args.plan)
    os.makedirs(args.out_dir, exist_ok=True)
    junit = os.path.abspath(os.path.join(args.out_dir, "junit.xml"))
    command = [
        "ctest", "--test-dir", args.build_dir, *ci_selection.selection_args(args.executor), "-j",
        str(args.jobs), "--no-tests=error", "--output-on-failure", "--output-junit", junit
    ]
    exit_code, metrics = ci_metrics.measure(command)
    last_test = os.path.join(args.build_dir, "Testing", "Temporary", "LastTest.log")
    if os.path.exists(last_test):
        shutil.copy(last_test, args.out_dir)
    report = _report(args, plan, junit, exit_code)
    report.update(_identity(args.bundle_sha256), metrics=metrics)
    _write_json(os.path.join(args.out_dir, "report.json"), report)
    return 1 if report["problems"] else 0


def cmd_preflight(args):
    """Refuse to run a shard whose bundle, selection or runtime is not the producer's."""
    plan = _read_json(args.plan)
    if plan["digest"] != args.plan_digest or ci_selection.plan_digest(plan) != plan["digest"]:
        raise ci_selection.SelectionError(
            f"plan {plan['digest']} is not the producer's {args.plan_digest}")
    ci_selection.check_restored(plan, args.build_dir, args.executor)
    ci_results.check_runtime(plan, _read_json(args.probes), ci_results.load_testing_tool())
    print(f"{args.executor}: {len(_expected(plan, args.executor))} tests match the producer; "
          f"{_tool_version('ctest')}")
    return 0


def cmd_measure(args):
    command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    exit_code, record = ci_metrics.measure(command)
    _write_json(args.out, record)
    return exit_code


def _tool_version(tool):
    return subprocess.run([tool, "--version"], check=True, capture_output=True,
                          text=True).stdout.splitlines()[0]


def cmd_bundle(args):
    plan = _read_json(args.plan)
    metadata = {
        "checkout_sha":
        subprocess.run(["git", "-C", args.workspace, "rev-parse", "HEAD"],
                       check=True,
                       capture_output=True,
                       text=True).stdout.strip(),
        "ctest":
        _tool_version("ctest"),
        "image":
        f"{os.environ.get('ImageOS', '')} {os.environ.get('ImageVersion', '')}".strip(),
        **_identity(None),
    }
    sha256 = ci_bundle.create(args.workspace, args.build_dir, plan, metadata, args.out)
    print(f"bundle {args.out}: sha256 {sha256}, plan {plan['digest']}")
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as fp:
            fp.write(f"bundle-sha256={sha256}\nplan-digest={plan['digest']}\n")
            fp.write(f"shard-executors={json.dumps(ci_selection.shard_executors())}\n")
    return 0


def cmd_restore(args):
    meta = ci_bundle.restore(args.tar, args.sha256, args.workspace)
    print(f"restored plan {meta['plan_digest']} built at {meta['checkout_sha']}")
    return 0


def _run_problems(args, consumers):
    """Problems of the run's jobs and of the producer's and consumers' reports."""
    reports = ci_gate.latest_reports(args.results_dir)
    problems = ci_gate.job_problems(json.loads(args.needs), args.required.split(","))
    if not args.plan_digest or not args.bundle_sha256:
        problems.append("the producer published no plan digest or bundle digest")
    problems += ci_gate.report_problems(reports, ["producer"], args.plan_digest, None)
    problems += ci_gate.report_problems(reports, consumers, args.plan_digest, args.bundle_sha256)
    return reports, problems


def _publish(path, title, body, problems):
    for problem in problems:
        print(f"::error::{problem}")
    verdict = "passed" if not problems else f"failed ({len(problems)} problems)"
    lines = [f"## {title}: {verdict}", "", body, ""] + [f"- {p}" for p in problems[:200]]
    if path:
        with open(path, "a", encoding="utf-8") as fp:
            fp.write("\n".join(lines) + "\n")
    return 1 if problems else 0


def cmd_compare(args):
    """The manual comparison: the monolithic baseline against the shards, same bundle."""
    shards = ci_selection.shard_executors()
    reports, problems = _run_problems(args, shards + ["baseline"])
    if "baseline" in reports and all(e in reports for e in shards):
        problems += ci_gate.compare_outcomes(reports["baseline"], [reports[e] for e in shards])
    return _publish(args.summary, "Linux regression comparison", ci_gate.timing_table(reports),
                    problems)


def cmd_summary(args):
    """linux-test-summary: the required check of the Linux PR tests."""
    args.required = ",".join(ci_gate.REQUIRED_JOBS)
    reports, problems = _run_problems(args, ci_selection.shard_executors())
    return _publish(args.summary, "Linux tests", ci_gate.timing_table(reports), problems)


def _add(commands, name, handler, help_text, *options):
    sub = commands.add_parser(name, help=help_text)
    for option in options:
        sub.add_argument(f"--{option}", required=True)
    sub.set_defaults(run=handler)
    return sub


def parser():
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest="command", required=True)
    _add(commands, "plan", cmd_plan, "write the executors' expected tests", "build-dir", "out")
    _add(commands, "probe", cmd_probe, "record the harness's dynamic capabilities", "plan", "out")
    evaluate = _add(commands, "evaluate", cmd_evaluate, "judge a JUnit file and exit code", "plan",
                    "probes", "executor", "junit", "out")
    evaluate.add_argument("--exit-code", type=int, required=True)
    run = _add(commands, "run", cmd_run, "run and judge one executor", "build-dir", "plan",
               "probes", "executor", "out-dir")
    run.add_argument("--jobs", type=int, default=2)
    run.add_argument("--bundle-sha256")
    measure = _add(commands, "measure", cmd_measure, "run a command, sampling memory", "out")
    measure.add_argument("argv", nargs=argparse.REMAINDER)
    bundle = _add(commands, "bundle", cmd_bundle, "package the regression bundle", "workspace",
                  "build-dir", "plan", "out")
    bundle.add_argument("--github-output")
    _add(commands, "restore", cmd_restore, "check and extract the bundle", "tar", "sha256",
         "workspace")
    compare = _add(commands, "compare", cmd_compare, "compare the baseline with the shards",
                   "results-dir", "needs", "required", "plan-digest", "bundle-sha256")
    compare.add_argument("--summary")
    summary = _add(commands, "summary", cmd_summary, "judge the Linux PR tests", "needs",
                   "results-dir")
    summary.add_argument("--plan-digest", default="")
    summary.add_argument("--bundle-sha256", default="")
    summary.add_argument("--summary")
    _add(commands, "preflight", cmd_preflight, "check a restored bundle before a shard runs",
         "build-dir", "plan", "plan-digest", "probes", "executor")
    return top


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return args.run(args)
    except (ci_selection.SelectionError, ci_bundle.BundleError,
            ci_results.RuntimeMismatch) as error:
        print(f"::error::{error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
