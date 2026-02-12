# Regression: set() from string iterable
if __name__ == "__main__":
    s = set("abca")
    assert len(s) == 3
    t = set("aAa".lower())
    assert len(t) == 1
    print("ok")
