# A name rebound through `global` must not fold to its first binding (#7945).
s = "big"
def setit() -> None:
    global s
    s = "little"
setit()
a = bytes([1, 0])
assert int.from_bytes(a, s) == 256
