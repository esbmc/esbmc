from modstub import C

# obj.v is 1 but the module global TAG is 2, so check() must fail. This guards
# against the named-imported global being silently dropped or made nondet: a
# dropped/nondet TAG would not deterministically refute self.v == TAG.
obj: C = C(1)
obj.check()
