def nondet_bool() -> bool: ...

# __python_bool_to_str: 6-byte buffer; True->"True", False->"False".
# Both branches covered with a nondet bool.
b = nondet_bool()
s = str(b)
if b:
    assert s == "True"
    assert len(s) == 4
else:
    assert s == "False"
    assert len(s) == 5
