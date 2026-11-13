# Elements of a list from _sv_verifier.nondet_list, called through its module,
# keep their type through nested comprehensions.
import _sv_verifier

ints = _sv_verifier.nondet_list(_sv_verifier.nondet_int)
doubled = [b + 1 for b in [a + 1 for a in ints]]
for v in doubled:
    assert isinstance(v, int)
