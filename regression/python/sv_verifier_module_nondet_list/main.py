# A harness that imports the module and calls _sv_verifier.nondet_list(...)
# through it must get a list of the generator type, as the
# `from _sv_verifier import nondet_list` spelling already does. Before, the
# element type was lost and isinstance() on each element failed.
import _sv_verifier

ints = _sv_verifier.nondet_list(_sv_verifier.nondet_int)
for v in ints:
    assert isinstance(v, int)

copied = []
for b in ints:
    copied.append(b + 1)
for v in copied:
    assert isinstance(v, int)
