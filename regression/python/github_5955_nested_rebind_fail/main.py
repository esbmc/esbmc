# The rebind sits inside an if-body: a top-level-only write scan counts one
# write and seeds the stale [1, 2] (GitHub #5955).
def t() -> bool:
    return True
l = [1, 2]
if t():
    l = [3, 3, 3]
assert l.count(3) == 0
