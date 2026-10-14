# Negative companion to github_4807_swapcase_param: wrong expected result
# must reach the solver and fail (rather than aborting in conversion).


def flip_case(s: str) -> str:
    return s.swapcase()


if __name__ == "__main__":
    assert flip_case('Hello') == 'WRONG'  # must fail
