# Negative companion to github_4807_empty_str_append: the crash fix exposed
# a real verification path, so an assertion that contradicts the result
# must now reach the solver and fail (rather than aborting in GOTO conversion).

def f():
    result = []
    result.append("")
    return result


if __name__ == "__main__":
    a = f()
    assert len(a) == 99  # must fail: list has exactly one element
