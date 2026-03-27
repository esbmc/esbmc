# Regression test for GitHub issue #3860:
# ESBMC reports false positive "invalid pointer dereference" when converting
# a non-recursive generator to a list via list(gen()).

def gen():
    yield 1
    yield 2
    yield 3

result = list(gen())
assert result == [1, 2, 3]
assert len(result) == 3
assert result[0] == 1
assert result[1] == 2
assert result[2] == 3
