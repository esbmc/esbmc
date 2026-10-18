# s may equal "b", when c is false.
s = nondet_str()
c = nondet_bool()
if s == ("a" if c else "b"):
    assert s == "a"
