# len() applied directly to an inline string-returning call whose result is a
# constant char array -- e.g. "{}".format(x) or s.replace(...) -- previously
# reported a spurious "array bounds violated" in strlen: the materialized temp
# was declared but never assigned, so strlen ran over uninitialised bytes.
assert len("{}".format("hi")) == 2
assert len("{:d}".format(65)) == 2
assert len("{}-{}".format("ab", "cd")) == 5
assert len("hi".replace("h", "H")) == 2
