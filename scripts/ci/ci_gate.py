"""Judge the jobs and reports of one workflow run.

A report only counts when its job succeeded in the run's latest attempt and
it carries the producer's plan digest and bundle digest: a green report from
an older attempt or another build never stands in for the current one.
"""

import json
import os
import re

_RESULT_DIR = re.compile(r"^linux-result-(?P<executor>.+)-(?P<attempt>\d+)$")
# Jobs linux-test-summary requires on a PR.
REQUIRED_JOBS = ("linux-build", "linux-regression", "cpython-check")


def job_problems(needs, required):
    """Required jobs that did not succeed; `needs` is the toJSON(needs) context."""
    problems = []
    for job in required:
        result = needs.get(job, {}).get("result", "missing")
        if result != "success":
            problems.append(f"job {job} is {result}")
    return problems


def latest_reports(results_dir):
    """executor -> report of its latest attempt among the downloaded artifacts."""
    latest = {}
    for entry in sorted(os.listdir(results_dir)):
        match = _RESULT_DIR.match(entry)
        if not match:
            continue
        attempt = int(match["attempt"])
        for root, _, files in os.walk(os.path.join(results_dir, entry)):
            if "report.json" in files:
                with open(os.path.join(root, "report.json"), encoding="utf-8") as fp:
                    report = json.load(fp)
                previous = latest.get(report["executor"])
                if previous is None or previous[0] < attempt:
                    latest[report["executor"]] = (attempt, report)
    return {executor: report for executor, (_, report) in latest.items()}


def report_problems(reports, executors, plan_digest, bundle_sha256):
    """Missing reports, reports of another build, and the problems reports hold."""
    problems = []
    for executor in executors:
        report = reports.get(executor)
        if report is None:
            problems.append(f"{executor} published no report")
            continue
        if report["plan_digest"] != plan_digest:
            problems.append(f"{executor} ran plan {report['plan_digest']}, not {plan_digest}")
        if bundle_sha256 and report.get("bundle_sha256") != bundle_sha256:
            problems.append(f"{executor} ran bundle {report.get('bundle_sha256')}, "
                            f"not {bundle_sha256}")
        problems += [f"{executor}: {p}" for p in report["problems"]]
    return problems


def timing_table(reports):
    rows = [
        "| executor | tests | CTest seconds | max sampled memory (MiB) | oom kills |",
        "|---|---|---|---|---|"
    ]
    for executor in sorted(reports):
        metrics = reports[executor].get("metrics", {})
        memory = metrics.get("memory_max_sampled")
        rows.append(f"| {executor} | {reports[executor]['expected']} | "
                    f"{metrics.get('seconds', 'n/a')} | "
                    f"{round(memory / 2**20) if memory else 'n/a'} | "
                    f"{metrics.get('oom_kills', 'n/a')} |")
    return "\n".join(rows)
