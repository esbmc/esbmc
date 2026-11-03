"""CTest selections of the Linux PR regression, and proof that they partition it.

The PR baseline is `ctest -LE python-intensive`. The producer runs the tests
without the `regression` label; the regression is split into modules of
suites, and the heaviest modules are striped into shards with `ctest -I i,,k`.
"""

import hashlib
import json
import re
import subprocess

INTENSIVE_LABEL = "python-intensive/"
REGRESSION_LABEL = "regression"
PYTHON_BENCHMARK_LABELS = ("quixbugs/", "humaneval/", "mopsa/", "consensus/", "ai-generated-code/",
                           "python-coverage/", "python-contracts/")
# Every C++ suite: esbmc-cpp/<suite>/ and esbmc-cpp11/<suite>/ to esbmc-cpp23/<suite>/.
CPP_LABELS = "^esbmc-cpp[0-9]*/.*$"

# Properties that make a test depend on another one or on a shared resource;
# striping assumes every test runs on its own, so any of them needs a review.
UNSUPPORTED_PROPERTIES = frozenset({
    "DEPENDS", "FIXTURES_CLEANUP", "FIXTURES_REQUIRED", "FIXTURES_SETUP", "PROCESSORS",
    "RESOURCE_GROUPS", "RESOURCE_LOCK", "RUN_SERIAL"
})

_LABEL_SYNTAX = re.compile(r"^[A-Za-z0-9_./-]+$")


class SelectionError(Exception):
    pass


def anchored(labels):
    """One CTest regex matching exactly these labels.

    Several -LE options are ANDed by CTest, so alternatives must share one
    regex. The labels hold no regex metacharacters, which is checked here.
    """
    for label in labels:
        if not _LABEL_SYNTAX.match(label):
            raise SelectionError(f"label {label!r} has characters a CTest regex would read")
    return "^(" + "|".join(labels) + ")$"


# (module, regex of its suite labels, shards), sized from the PR job's
# timings (docs/ci/linux-regression.md). The last module has no regex: it
# takes every other regression test, so a new suite is never left out.
MODULES = (
    ("numpy", anchored(["numpy/"]), 2),
    ("python", anchored(["python/"]), 2),
    ("python-benchmarks", anchored(PYTHON_BENCHMARK_LABELS), 1),
    ("cpp", CPP_LABELS, 1),
    ("c-and-tools", None, 1),
)


def _module_args(regex):
    if regex is not None:
        return ["-L", regex, "-LE", anchored([INTENSIVE_LABEL])]
    others = [r for _, r, _ in MODULES if r] + [anchored([INTENSIVE_LABEL])]
    return ["-L", anchored([REGRESSION_LABEL]), "-LE", "|".join(others)]


def _shard_args():
    """executor -> CTest options, one executor per shard of each module."""
    table = {}
    for name, regex, shards in MODULES:
        args = _module_args(regex)
        if shards == 1:
            table[name] = args
            continue
        for i in range(1, shards + 1):
            table[f"{name}-{i}"] = args + ["-I", f"{i},,{shards}"]
    return table


def shard_executors():
    return list(_shard_args())


def module_of(labels):
    """The module whose shards run a regression test with these labels."""
    for name, regex, _ in MODULES:
        if regex is None or any(re.match(regex, label) for label in labels):
            return name
    raise SelectionError(f"no module for labels {labels}")


def selection_args(executor):
    """The CTest options that select one executor's tests."""
    if executor == "baseline-pr":
        return ["-LE", anchored([INTENSIVE_LABEL])]
    if executor == "producer":
        return ["-LE", anchored([REGRESSION_LABEL])]
    if executor == "baseline":
        return ["-L", anchored([REGRESSION_LABEL]), "-LE", anchored([INTENSIVE_LABEL])]
    table = _shard_args()
    if executor not in table:
        raise SelectionError(f"unknown executor {executor!r}")
    return table[executor]


def show_only(build_dir, args):
    """The tests CTest selects with `args`, as its json-v1 entries."""
    completed = subprocess.run(
        ["ctest", "--test-dir", str(build_dir), "--show-only=json-v1", *args],
        check=True,
        capture_output=True,
        text=True)
    return json.loads(completed.stdout)["tests"]


def properties(test):
    return {p["name"]: p["value"] for p in test.get("properties", [])}


def test_record(test):
    """The parts of a test that must not change between producer and consumer."""
    props = properties(test)
    return {
        "command": test.get("command", []),
        "labels": props.get("LABELS", []),
        "environment": props.get("ENVIRONMENT", []),
        "timeout": props.get("TIMEOUT"),
        "skip_return_code": props.get("SKIP_RETURN_CODE"),
        "working_directory": props.get("WORKING_DIRECTORY"),
    }


def _unique_names(tests, what):
    names = [t["name"] for t in tests]
    seen = set()
    duplicates = sorted({n for n in names if n in seen or seen.add(n)})
    if duplicates:
        raise SelectionError(f"{what} repeats test names: {duplicates[:5]}")
    if not names:
        raise SelectionError(f"{what} selects no test")
    return names


def check_inventory(inventory):
    """Fail on what the striping model cannot represent."""
    _unique_names(inventory, "the inventory")
    for test in inventory:
        props = properties(test)
        labels = props.get("LABELS", [])
        if INTENSIVE_LABEL in labels and REGRESSION_LABEL not in labels:
            raise SelectionError(f"{test['name']} is python-intensive without the regression "
                                 "label, so the producer would run it")
        unsupported = UNSUPPORTED_PROPERTIES.intersection(props)
        if unsupported:
            raise SelectionError(
                f"{test['name']} sets {sorted(unsupported)}, which shard striping ignores")


def check_partition(baseline, executors, parts):
    """`parts` (executor -> names) must cover `baseline` once, and nothing else."""
    owner = {}
    for executor in parts:
        if not parts[executor]:
            raise SelectionError(f"{executor} selects no test")
        for name in parts[executor]:
            if name in owner:
                raise SelectionError(f"{name} is selected by {owner[name]} and {executor}")
            owner[name] = executor
    missing = sorted(set(baseline) - set(owner))
    extra = sorted(set(owner) - set(baseline))
    if missing or extra:
        raise SelectionError(f"the executors miss {missing[:5]} and add {extra[:5]} "
                             f"({len(missing)} missing, {len(extra)} extra) against "
                             f"{', '.join(executors)}")


def make_plan(build_dir):
    """The expected tests of every executor, proven to partition the PR baseline."""
    inventory = show_only(build_dir, [])
    check_inventory(inventory)
    records = {t["name"]: test_record(t) for t in inventory}
    baseline = _unique_names(show_only(build_dir, selection_args("baseline-pr")), "the baseline")
    executors = ["producer"] + shard_executors()
    parts = {e: _unique_names(show_only(build_dir, selection_args(e)), e) for e in executors}
    check_partition(baseline, executors, parts)
    regression = _unique_names(show_only(build_dir, selection_args("baseline")), "baseline")
    check_partition(regression, shard_executors(), {e: parts[e] for e in shard_executors()})
    plan = {
        "schema_version": 1,
        "executors": parts,
        "baseline": regression,
        "tests": {
            name: records[name]
            for name in baseline
        },
    }
    plan["digest"] = plan_digest(plan)
    return plan


def plan_digest(plan):
    body = {k: v for k, v in plan.items() if k != "digest"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def check_restored(plan, build_dir, executor):
    """The restored build must select exactly what the producer planned."""
    expected = plan["baseline"] if executor == "baseline" else plan["executors"][executor]
    tests = show_only(build_dir, selection_args(executor))
    names = _unique_names(tests, executor)
    if names != expected:
        raise SelectionError(f"{executor} selects {len(names)} tests on this runner, "
                             f"{len(expected)} on the producer")
    for test in tests:
        if test_record(test) != plan["tests"][test["name"]]:
            raise SelectionError(f"{test['name']} differs from the producer's inventory")
