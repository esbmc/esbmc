def process(val):
    if isinstance(val, type(None)):
        return 0
    else:
        return val

result = process(None)
assert result == 0
