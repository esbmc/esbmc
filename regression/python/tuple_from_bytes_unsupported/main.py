# tuple(bytes) would be a tuple of ints in CPython (tuple(b"ab") == (97, 98)),
# which is not modelled. It must be rejected with a clean error rather than
# silently reusing the tuple(str) char-string lowering (a wrong verdict).
t = tuple(b"ab")
