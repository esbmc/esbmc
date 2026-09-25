# The unpacked components are really read, not assumed.
pairs = [(1, 2), (3, 4)]
for i, tpl in enumerate(pairs):
    a, b = tpl
    assert a > b
