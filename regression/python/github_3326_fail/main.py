def foo() -> dict[str, list[str]]:
    return {
        "fruit": ["apple", "banana"],
        "qux": ["quux", "quuz"]
    }

f = foo()
l = f["fruit"]
for s in l:
    assert s == "wrong"

