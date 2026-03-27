# Regression: set() from string parameter


def f(s: str) -> int:
    return len(set(s.lower()))


if __name__ == "__main__":
    assert f("aAa") == 1
    assert f("abc") == 3
    print("ok")
