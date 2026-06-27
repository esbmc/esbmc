# list(bytes) would be a list of ints in CPython (list(b"ab") == [97, 98]),
# which is not modelled. It must be rejected with a clean error rather than
# silently reusing the list(str) char-string lowering (a wrong verdict).
x = list(b"ab")
