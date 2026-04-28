def main():
    d = {"a": 1, "b": 2}
    assert set(d.keys()) == {"b", "a"}
    assert set(d.values()) == {2, 1}
    assert set(d.keys()) != {"a"}


main()
