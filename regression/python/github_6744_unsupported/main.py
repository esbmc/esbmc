def build() -> list[tuple]:
    return [(1, 2)]


pairs = build()
for i, (a, b) in enumerate(pairs):
    assert a + b == 3
