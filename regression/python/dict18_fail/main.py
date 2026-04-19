def foo() -> dict[str, list[str]]:
    return {
        "foo": ["bar", "baz"],
        "qux": ["quux", "quuz"]
    }

f = foo()
l:list[str] = f["foo"]
assert len(l) == 2
for s in l:
    assert s == "bar"
