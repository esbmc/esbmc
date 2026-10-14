# len() of a dict-comprehension result. The comprehension assigns the
# __python_dict__ struct directly to the variable without recording a "dict"
# annotation, so len() used to fall through to strlen() and read the struct's
# bytes as a C string, yielding a wrong size. len() must instead route to the
# dict-size handler. Regression for #5222 (the len() portion).
d = {i: i for i in range(2)}
assert len(d) == 2
# Read-back still works alongside the corrected len().
assert d[1] == 1
