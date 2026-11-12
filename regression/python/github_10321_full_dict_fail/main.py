import _sv_verifier


def main() -> None:
    d = _sv_verifier.nondet_dict(_sv_verifier.nondet_int, _sv_verifier.nondet_int)
    assert len(d) != 8


main()
