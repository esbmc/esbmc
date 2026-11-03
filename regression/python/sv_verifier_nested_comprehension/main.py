# f is called on the ints of xs and on its own results, so its parameter is
# an int. The second call site resolves only once f is typed.
from _sv_verifier import nondet_list, nondet_int


def f(a):
    return a + 1


xs = nondet_list(nondet_int)
c = [f(a) for a in [f(b) for b in xs]]
for v in c:
    assert isinstance(v, int)
