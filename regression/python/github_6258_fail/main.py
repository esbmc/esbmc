# github.com/esbmc/esbmc/issues/6258
# Same builtin-exception handling, but the asserted outcome is wrong: the
# except block runs, so `caught` is True and `assert not caught` must fail.
caught = False
try:
    raise KeyboardInterrupt("stop")
    caught = False
except KeyboardInterrupt:
    caught = True
assert not caught
