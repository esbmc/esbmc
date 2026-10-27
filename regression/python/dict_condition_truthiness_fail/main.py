# A dict with a key is truthy as a condition, so the assertion fails.
def classify(d: dict[str, int]) -> int:
    if d:
        return 1
    return 0


flag = nondet_bool()
full = {"a": 1}
if full or flag:
    assert classify(full) == 0
