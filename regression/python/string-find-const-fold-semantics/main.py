s = "hello world"

assert s.find("o") == 4
assert s.rfind("o") == 7
assert s.find("zz") == -1
assert s.rfind("zz") == -1
assert s.index("world") == 6
assert s.rindex("l") == 9
assert s.find("") == 0
assert s.rfind("") == 11
