# Even an unknown receiver is unchanged by a nonpositive width.
s: str = nondet_str()
n: int = nondet_int()
if s == "ab" and -3 <= n <= -1:
    assert s.center(n) == s
    assert s.ljust(n) == s
    assert s.rjust(n) == s
    assert s.zfill(n) == s
