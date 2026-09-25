# github.com/esbmc/esbmc/issues/6258
# KeyboardInterrupt is a builtin exception; referencing/catching it must not
# raise a spurious NameError.
caught = False
try:
    raise KeyboardInterrupt("stop")
    caught = False
except KeyboardInterrupt:
    caught = True
assert caught
