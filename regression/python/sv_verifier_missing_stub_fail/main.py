# No _sv_verifier.py next to this file: ESBMC falls back to its bundled stub.
import _sv_verifier
from _sv_verifier import nondet_bool

x = _sv_verifier.nondet_int()
b = nondet_bool()
if b:
    assert x > 0
