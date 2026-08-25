def run():
    d = {"a": 1, "b": 2}
    s = "hello"
    xs = [10, 20, 30]
    q = 9 // 3
    assert d["a"] + d["b"] + xs[2] + q == 36
    assert s[1] == "e"
    assert xs.index(20) == 1


def main():
    run()


if __name__ == "__main__":
    main()
