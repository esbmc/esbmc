from stubs import *

t: Container = Container()
t[0, :, :] = make_tile(3, 4)
assert t.value.d0 == 3
