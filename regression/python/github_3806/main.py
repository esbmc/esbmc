# Minimal reproducer from issue #3806
# list(d.items()) on an empty dict should equal []
d = {}
assert list(d.items()) == []
