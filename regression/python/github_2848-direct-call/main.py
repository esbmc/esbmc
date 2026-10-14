from typing import Any


def pick(c: bool) -> Any:
    if c:
        return 10
    return "ten"


def main() -> None:
    # Control for github_2848: asserting on the call directly already works,
    # which locates the defect in the assignment path rather than in the
    # return-type inference itself.
    assert pick(False) == "ten"


main()
