# This file is part of the SV-Benchmarks collection of verification tasks:
# https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
#
# SPDX-FileCopyrightText: 2026 Mostafa Hassan, Simon Wehrli, Caterina Urban, Marco Eilers
#
# SPDX-License-Identifier: MPL-2.0

import _sv_verifier

def f1(x):
    u = x[0]
    v = x * 3
    return u * v

def f2(x):
    return [x]

def f3(x):
    return x

def f4(x, y):
    u = x["string"] + y
    v = y + 3
    return u | v

def f5(x, y, i):
    t = x * 3
    u = y["string"]
    v = x[i]
    return (t[0] + v) * (u + 3)

def f6(x):
    return x["string"]

def f7(x):
    if x < 2:
        return 1
    return f7(x - 1) + f7(x - 2)

a = f1([1, 2, 3])
b = f2(a[0])
c = f3(b)
d = f4({"string": c[0]}, 1)
e = f5(a, {"string": d}, 2)
f = b[e - e]
g = f6({"string": 1})
h = b[g - g]
_sv_verifier.check_type(a, list[int])
_sv_verifier.check_type(b, list[int])
_sv_verifier.check_type(c, list[int])
_sv_verifier.check_type(d, int)
_sv_verifier.check_type(e, int)
_sv_verifier.check_type(f, int)
_sv_verifier.check_type(g, int)
_sv_verifier.check_type(h, int)
