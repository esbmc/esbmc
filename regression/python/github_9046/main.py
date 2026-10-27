# Same-named functions of two imported modules stay apart (#9046)
import one
import two

assert one.f() == 1
assert two.f() == 2
