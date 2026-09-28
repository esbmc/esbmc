caught = False
try:
    raise KeyboardInterrupt("stop")
    caught = False
except KeyboardInterrupt:
    caught = True
assert not caught
