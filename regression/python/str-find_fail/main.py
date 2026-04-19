s: str = "foo:bar:baz"
i = s.find(':')
assert i != 3

i1 = s.find('')
assert i1 == 0

i2 = s.find('d')
assert i2 == -1
