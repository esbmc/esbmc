# Importing SV-COMP's _sv_verifier stub module used to crash ESBMC with a core
# dump: converting the body of nondet_float (a function-local `import sys` plus
# sys.float_info) hit an undefined name and abort()ed. The nondet_* scalar stubs
# are intercepted at their call sites, so their bodies must not be converted.
# Here the float value is constrained away from the failing region, so the
# assertion holds: VERIFICATION SUCCESSFUL.
from _sv_verifier import nondet_int, nondet_float, nondet_str


def main() -> None:
    x = nondet_int()
    if x > 0:
        assert x > 0

    v = nondet_float()
    if v > 0.0:
        assert v > 0.0

    s = nondet_str()
    assert s is not None


main()
