# A starred dict passes its keys, which indexing by position does not give,
# so the call is not spelled out.
def cat(a, b):
    return a + b


def call(d):
    return cat(*d)


assert call({"x": 1, "y": 2}) == "xy"
