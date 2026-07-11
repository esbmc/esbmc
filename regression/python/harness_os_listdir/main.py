# Verification harness for os.listdir (src/python-frontend/models/os.py).
#
# The os model returns a fixed two-entry directory listing ["foo", "bar"], a
# deterministic stand-in for filesystem contents under verification. The model
# deliberately diverges from CPython's os.listdir, so this test is verified
# with ESBMC only (excluded from the CPython sanity sweep in
# scripts/check_python_tests.sh).
#
# ENSURES:
#   E1: listdir returns two entries
#   E2: the entries are "foo" and "bar" in order
import os

entries: list[str] = os.listdir("/some/path")

assert len(entries) == 2          # E1
assert entries[0] == "foo"        # E2
assert entries[1] == "bar"        # E2
