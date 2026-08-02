# A parameter annotated with a bare ``dict``/``Dict`` (no key/value types) must
# still be recognised as a dict, so ``sorted(d)`` / ``list(d)`` iterate its keys
# instead of reinterpreting the dict struct as a list (which read a wrong length
# and tripped a spurious out-of-bounds failure). See GitHub #4790 / #4803.
from typing import Dict


def n_via_sorted(d: dict) -> int:
    ks = sorted(d)
    return len(ks)


def n_via_list(d: Dict) -> int:
    return len(list(d))


def main() -> None:
    m = {1: 10, 2: 20, 3: 30}
    assert n_via_sorted(m) == 3
    assert n_via_list(m) == 3


main()
