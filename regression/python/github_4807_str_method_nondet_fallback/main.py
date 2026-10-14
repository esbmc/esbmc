# Regression for #4807: str methods that previously aborted on a
# non-constant receiver (swapcase / isupper / isalnum / count / capitalize
# / title / casefold) now fall back to a sound symbolic value, so GOTO
# conversion succeeds. Verifies the wrapper functions all convert and
# execute (the symbolic results are unconstrained by design).


def f_swapcase(s: str) -> str:
    return s.swapcase()


def f_isupper(s: str) -> bool:
    return s.isupper()


def f_count(s: str, c: str) -> int:
    return s.count(c)


def f_isalnum(s: str) -> bool:
    return s.isalnum()


def f_capitalize(s: str) -> str:
    return s.capitalize()


def f_title(s: str) -> str:
    return s.title()


def f_casefold(s: str) -> str:
    return s.casefold()


if __name__ == "__main__":
    # Constant-arg call sites are consteval-folded.
    assert f_swapcase('Hi') == 'hI'
    assert f_isupper('AB') == True
    assert f_isupper('aB') == False
    assert f_count('aabba', 'a') == 3
    assert f_isalnum('abc') == True
    assert f_capitalize('hello') == 'Hello'
    assert f_title('hello world') == 'Hello World'
    assert f_casefold('AB') == 'ab'
