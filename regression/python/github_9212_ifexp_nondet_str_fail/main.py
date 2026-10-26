# A conditional expression choosing a str literal or nondet_str() (#9212).
c = nondet_bool()
k = "a" if c else nondet_str()
if not c:
    assert len(k) == 1
