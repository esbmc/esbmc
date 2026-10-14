# Integration guard for the str runtime-model dispatch paths whose silent
# nondet fallback #4827 makes observable (capitalize / title / swapcase /
# count / isalnum / isupper). Each receiver is passed through a wrapper so it
# is NON-constant at conversion time -- this forces the handler to dispatch to
# the __python_str_* operational model (rather than consteval-folding the
# literal), which is exactly the path #4827 guards. The exact-value assertions
# only hold if the runtime models actually compute the result.


def f_capitalize(s: str) -> str:
    return s.capitalize()


def f_title(s: str) -> str:
    return s.title()


def f_swapcase(s: str) -> str:
    return s.swapcase()


def f_count(s: str, c: str) -> int:
    return s.count(c)


def f_isalnum(s: str) -> bool:
    return s.isalnum()


def f_isupper(s: str) -> bool:
    return s.isupper()


if __name__ == "__main__":
    assert f_capitalize("hELLO") == "Hello"
    assert f_title("hello world") == "Hello World"
    assert f_swapcase("Hello") == "hELLO"
    assert f_count("aabba", "a") == 3
    assert f_isalnum("abc123") == True
    assert f_isalnum("ab c") == False
    assert f_isupper("ABC") == True
    assert f_isupper("aBC") == False
