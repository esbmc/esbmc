# A `type` parameter shares the char-array model with str, so isinstance must
# not decide it as a string would: f(int) makes the assertion fail.
def f(t: type) -> None:
    assert not isinstance(t, type)


f(int)
