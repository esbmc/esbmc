# Regression test for GitHub issue #3860 (fail variant):
# Verifies that incorrect assertions on list(gen()) are caught.


def gen():
    yield 1
    yield 2
    yield 3


result = list(gen())
# This assertion is wrong: list has 3 elements, not 2.
assert len(result) == 2
