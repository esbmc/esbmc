# Same #4830 root cause through a dict literal: call-site inference must
# produce dict[str, list[int]] so the nested list value resolves to int.
def g(d, x):
    return x - d["a"][0]


assert g({"a": [3]}, 1) == -2
