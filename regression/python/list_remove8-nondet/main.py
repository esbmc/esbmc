b: bool = nondet_bool()

l = ["hello", "world", "hello"]

if b:
    l.remove("hello")

    assert len(l) == 2

    # First hello removed
    assert l[0] == "world"
    assert l[1] == "hello"
else:
    assert len(l) == 3
