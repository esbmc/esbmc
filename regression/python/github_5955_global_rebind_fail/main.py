# f() rebinds x via `global` with no module-level assignment statement, so
# the write-once seed x=1 is stale and must not fold (GitHub #5955).
x = 1
def f() -> None:
    global x
    x = 2
def g(v: int) -> int:
    return v
f()
assert g(x) == 1
