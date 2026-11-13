# e is rebound before the raise, so the ValueError escapes, not the IndexError.
l = [1, 2]
try:
    x = l[5]
except Exception as e:
    e = ValueError("rebound")
    raise e
