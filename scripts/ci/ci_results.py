"""What one CTest run reported, and whether it proves its tests passed.

The JUnit test cases, the selection planned by the producer and the CTest
exit code are the authority; the aggregate JUnit counters are not, since CTest
counts a test whose executable is missing as skipped.
"""

import hashlib
import importlib.util
import os
import shutil
import xml.etree.ElementTree as ET

PASSED, SKIPPED, FAILED, INVALID = "passed", "skipped", "failed", "invalid"
SKIP_MESSAGE = "SKIP_RETURN_CODE=10"

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_testing_tool():
    """The regression harness, so a skip is explained by its own logic."""
    path = os.path.join(_REPO, "regression", "testing_tool.py")
    spec = importlib.util.spec_from_file_location("testing_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify_case(case):
    """One JUnit test case as passed, skipped (code 10), failed or invalid."""
    status = case.get("status")
    failures = case.findall("failure") + case.findall("error")
    skipped = case.findall("skipped")
    if failures:
        return FAILED, failures[0].get("message", "")
    if status == "run" and not skipped:
        return PASSED, ""
    if status == "notrun" and len(skipped) == 1 and skipped[0].get("message") == SKIP_MESSAGE:
        return SKIPPED, SKIP_MESSAGE
    message = skipped[0].get("message", "") if skipped else ""
    return INVALID, f"status={status!r} {message}".strip()


def read_junit(path):
    """name -> (outcome, message). A repeated name is itself an invalid result."""
    results = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        name = case.get("name")
        if name in results:
            results[name] = (INVALID, "reported more than once")
        else:
            results[name] = classify_case(case)
    return results


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def harness_arguments(command):
    """--regression, --file, --modes, --capabilities and --tool of a regression test."""
    args = {"modes": []}
    collecting_modes = False
    for arg in command:
        if arg.startswith("--"):
            collecting_modes = arg == "--modes"
            key, sep, value = arg[2:].partition("=")
            if sep:
                args[key] = value
        elif collecting_modes:
            args["modes"].append(arg)
    return args


def _static_capabilities(args, environment):
    if "capabilities" not in args:
        return None
    capabilities = {c for c in args["capabilities"].replace(";", ",").split(",") if c}
    timeout = next((int(e.split("=", 1)[1])
                    for e in environment if e.startswith("ESBMC_REGRESS_TIMEOUT=")), None)
    if timeout is not None and timeout < 600:
        capabilities.discard("long_timeout")
    return capabilities


def skip_reason(record, probes, harness):
    """Why the harness skips this test, or None when nothing explains a skip.

    Mirrors gen_one_test in regression/testing_tool.py: a mode outside
    --modes, or a REQUIRES capability absent from the static set or from the
    dynamic probes the producer recorded.
    """
    args = harness_arguments(record["command"])
    if "regression" not in args or "file" not in args:
        return None
    case = harness.TestCase(os.path.join(args["regression"], args["file"]), args["file"])
    if case.test_mode not in args["modes"]:
        return f"mode {case.test_mode} is not in {args['modes']}"
    static = _static_capabilities(args, record["environment"])
    tool_probes = probes.get(args.get("tool", ""), {})
    missing = [
        c for c in case.requires
        if (c in harness.DYNAMIC_CAPABILITY_PROBES and tool_probes.get(c) is False) or (
            c not in harness.DYNAMIC_CAPABILITY_PROBES and static is not None and c not in static)
    ]
    return f"requires {', '.join(missing)}" if missing else None


def probe_capabilities(plan, harness):
    """tool -> dynamic capability -> supported, as the harness would probe it here."""
    tools = {harness_arguments(r["command"]).get("tool") for r in plan["tests"].values()}
    # The harness's own probe, so the producer cannot disagree with it.
    return {
        tool: {
            c: harness._probe_capability(c, tool)  # pylint: disable=protected-access
            for c in sorted(harness.DYNAMIC_CAPABILITY_PROBES)
        }
        for tool in sorted(t for t in tools if t)
    }


class RuntimeMismatch(Exception):
    """The runner cannot run the tests as the producer's build expects."""


def check_runtime(plan, producer_probes, harness):
    """This runner must answer the harness's capability questions as the producer did.

    A difference would turn a test into a new skip, or a skip into a failure,
    for a reason that has nothing to do with the change under test.
    """
    here = probe_capabilities(plan, harness)
    if here != producer_probes:
        raise RuntimeMismatch(f"dynamic capabilities differ: producer {producer_probes}, "
                              f"this runner {here}")
    needs_z3 = any("z3_binary" in harness_arguments(r["command"]).get("capabilities", "")
                   for r in plan["tests"].values())
    if needs_z3 and not shutil.which("z3"):
        raise RuntimeMismatch("the build granted z3_binary, but no z3 is on PATH")


def evaluate_run(expected, junit_results, exit_code, plan, probes, harness):
    """Problems that keep a run from proving its tests passed; empty when it does."""
    problems = []
    if exit_code != 0:
        problems.append(f"ctest exited with {exit_code}")
    missing = [n for n in expected if n not in junit_results]
    extra = [n for n in junit_results if n not in set(expected)]
    if missing:
        problems.append(f"{len(missing)} selected tests have no JUnit result: {missing[:5]}")
    if extra:
        problems.append(f"{len(extra)} JUnit results were not selected: {extra[:5]}")
    for name in expected:
        outcome, message = junit_results.get(name, (PASSED, ""))
        if outcome in (FAILED, INVALID):
            problems.append(f"{name}: {outcome} {message}".rstrip())
        elif outcome == SKIPPED and not skip_reason(plan["tests"][name], probes, harness):
            problems.append(f"{name}: skipped with no reason the harness would give")
    return problems


def run_report(executor, expected, junit_path, exit_code, plan, probes, harness):
    """The verdict of one run, tied to its plan and its JUnit file."""
    report = {
        "executor": executor,
        "plan_digest": plan["digest"],
        "exit_code": exit_code,
        "expected": len(expected),
        "junit_digest": None,
        "outcomes": {},
        "problems": [],
    }
    if not os.path.exists(junit_path):
        report["problems"].append("no JUnit file")
        return report
    report["junit_digest"] = file_digest(junit_path)
    try:
        results = read_junit(junit_path)
    except ET.ParseError as error:
        report["problems"].append(f"incomplete JUnit: {error}")
        return report
    report["outcomes"] = {name: outcome for name, (outcome, _) in results.items()}
    report["problems"] = evaluate_run(expected, results, exit_code, plan, probes, harness)
    return report
