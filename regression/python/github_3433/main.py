def check(items: list) -> bool:
    return all(isinstance(x, str) for x in items)


result = check(["a", "b"])
assert result
