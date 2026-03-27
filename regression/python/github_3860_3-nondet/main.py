# Regression test for GitHub issue #3860 (variant 3):
# Generator with nondet conditional yield converted to list.

def nondet_bool() -> bool:
    pass

def gen_cond():
    yield 10
    if nondet_bool():
        yield 20
    yield 30

result = list(gen_cond())
# The list must start with 10 and end with 30; may or may not contain 20.
assert result[0] == 10
assert len(result) >= 2
