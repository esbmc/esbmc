# Companion to sv_verifier_nondet_stub: the same _sv_verifier stub module that
# used to crash ESBMC on import, now exercised so that a real property fails.
# nondet_float() may return NaN, and NaN != NaN, so the reflexive equality
# assertion does not hold: VERIFICATION FAILED.
from _sv_verifier import nondet_float


def main() -> None:
    v = nondet_float()
    assert v == v


main()
