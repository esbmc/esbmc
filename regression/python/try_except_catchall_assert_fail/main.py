# The catch-all handler runs and its assertion fails: ESBMC must report the
# violation rather than crash.
state = 0
try:
    raise ValueError("boom")
except:
    assert state == 1

assert False
