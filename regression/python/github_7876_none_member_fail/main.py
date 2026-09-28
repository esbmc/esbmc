# Counterpart to github_7876_none_member.
def f(y: int | list[int] | None) -> int:
    if y is None:
        return 0
    return y + 1


assert f(0) != 1
