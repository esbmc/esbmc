"""Shared scaffolding for the Tier-C oracle drivers.

Each oracle in this directory needs the same four things: ESBMC's argument list
for a regression test, a run that cannot hang or pollute a sibling, the verdict
its output ends with, and the triaged-divergence baseline. Keeping one copy is
not only tidier -- the argument list in particular has to match `ctest` exactly,
and three drifting copies of that is three ways to report a divergence that is
really an invocation difference.
"""

import os
import re
import shutil
import subprocess  # nosec B404 - argv lists only, never a shell; see capture()
import sys
import tempfile

REGRESSION = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "regression"
)
sys.path.insert(0, os.path.abspath(REGRESSION))

# pylint: disable=import-error,wrong-import-position
from testing_tool import TestCase  # type: ignore  # noqa: E402

VERDICT = re.compile(r"^VERIFICATION (SUCCESSFUL|FAILED|UNKNOWN)$", re.MULTILINE)

# A run that reached no verdict says nothing about the property under test. Both
# are reported by name rather than folded into agreement.
TIMEOUT = "timeout"
NO_VERDICT = "no-verdict"


def esbmc_path(parser, given):
    path = os.path.abspath(given)
    if not os.access(path, os.X_OK):
        parser.error(f"not executable: {path}")
    return path


def capture(esbmc, args, cwd, timeout):
    """ESBMC's combined output, or None if it did not finish in time.

    The command is a fixed argv list -- the verified ESBMC binary plus flags the
    caller assembled -- so there is no shell and no interpolation of external
    input; `esbmc_path` has already checked the binary is executable.
    """
    try:
        proc = subprocess.run(  # nosec B603 - argv list, shell=False
            [esbmc] + args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,  # a FAILED verdict exits non-zero and is a result
        )
    except subprocess.TimeoutExpired:
        return None
    return proc.stdout.decode("utf-8", "replace")


def verdict_of(text):
    """The run's verdict. Under --multi-property ESBMC prints one line per
    property, and the run failed if any property did."""
    if text is None:
        return TIMEOUT
    found = VERDICT.findall(text)
    if not found:
        return NO_VERDICT
    return "FAILED" if "FAILED" in found else found[-1]


def scratch_dir(prefix):
    return tempfile.mkdtemp(prefix=prefix)


def drop_scratch(path):
    shutil.rmtree(path, ignore_errors=True)


def collect_tests(suite, modes, exclude_options=()):
    """The regression tests of `suite` in `modes`, minus any whose own flags name
    an option the caller is about to drive itself.

    The suite path is absolutised because `TestCase` resolves the input file
    against it, and the oracles run ESBMC in a scratch cwd where a repo-relative
    path would not exist.
    """
    suite = os.path.abspath(suite)
    tests = []
    for entry in sorted(os.listdir(suite)):
        directory = os.path.join(suite, entry)
        if not os.path.isfile(os.path.join(directory, "test.desc")):
            continue
        case = TestCase(directory, entry)
        if case.test_mode not in modes:
            continue
        if any(opt in case.test_args for opt in exclude_options):
            continue
        tests.append(case)
    return tests


def load_baseline(path):
    """Test names already triaged as diverging. '#' starts a comment."""
    if not path:
        return set()
    names = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            name = line.split("#", 1)[0].strip()
            if name:
                names.add(name)
    return names


def report_baseline(baseline, diverged_names):
    """Print the baseline verdict and return the untriaged divergences.

    A listed test that now agrees is reported as stale: leaving the entry in
    would let a later regression of the same test pass unnoticed.
    """
    new = sorted(diverged_names - baseline)
    if baseline:
        print(f"\nbaselined    {len(diverged_names & baseline)} of {len(baseline)}")
        for name in sorted(baseline - diverged_names):
            print(f"STALE-BASELINE {name}: listed as diverging but agrees now")
        for name in new:
            print(f"NEW-DIVERGENCE {name}")
    return new
