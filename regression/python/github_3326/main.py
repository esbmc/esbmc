def foo() -> dict[str, list[str]]:
    return {
        "foo": ["bar", "baz"],
        "qux": ["quux", "quuz"]
    }

f = foo()
l = f["foo"]
for s in l:
    assert s == "bar" or s == "baz"

