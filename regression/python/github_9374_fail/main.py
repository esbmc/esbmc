# bool() of a str is true exactly when the str is non-empty (#9374).
r = bool("abc")
assert not r
