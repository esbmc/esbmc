# As github_9325_slice, but asserts the first element of fs.
def f():
    return 1


def g():
    return "a"


fs = [f, g, f]
sub = fs[1:]
r = sub[0]()
assert r == 1
