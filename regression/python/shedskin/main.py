# Simple runtime smoke test for Shedskin-generated runtime helpers

def test_list_ops():
    xs = [1, 2, 2, 3]
    assert xs.count(2) == 2
    assert xs.index(2) == 1
    xs.remove(2)
    assert xs == [1, 2, 3]
    try:
        xs.remove(99)
        raise AssertionError("remove should have failed")
    except Exception:
        pass


def test_dict_ops():
    d = {"a": 1, "b": 2}
    assert d.get("a", 0) == 1
    assert d.get("missing", 42) == 42
    k = sorted(list(d.keys()))
    v = sorted(list(d.values()))
    items = sorted(list(d.items()))
    assert k == ["a", "b"]
    assert v == [1, 2]
    assert items == [("a", 1), ("b", 2)]


def test_str_ops():
    s = "hello world"
    assert s.startswith("he")
    assert s.endswith("world")
    assert s.find("lo") == 3
    assert s.find("zz") == -1
    assert s.rfind("l") == 9


def test_set_ops():
    st = set([1, 2, 3])
    st.add(3)
    assert 2 in st
    st.discard(2)
    assert 2 not in st
    popped = st.pop()
    assert popped in [1, 3]
    try:
        while True:
            st.pop()
    except Exception:
        pass


def main():
    test_list_ops()
    test_dict_ops()
    test_str_ops()
    test_set_ops()
    print("shedskin runtime smoke: OK")


if __name__ == "__main__":
    main()
