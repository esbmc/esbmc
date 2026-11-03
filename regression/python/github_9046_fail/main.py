# Same-named functions of two imported modules stay apart (#9046)
import one
import two

assert two.f() == 1
