# This file is part of the SV-Benchmarks collection of verification tasks:
# https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
#
# SPDX-FileCopyrightText: 2026 Mostafa Hassan, Simon Wehrli, Caterina Urban, Marco Eilers
#
# SPDX-License-Identifier: MPL-2.0

import _sv_verifier

class A:
    def f(self):
        return B()

    def g(self):
        return self.f().g()

class B(A):
    def g(self):
        return "string"

x = A()
y = x.g()

_sv_verifier.check_type(A, type)
_sv_verifier.check_type(B, type)
_sv_verifier.check_type(x, A)
_sv_verifier.check_type(y, str)

# A := Type[A]
# B := Type[B]
# x := A
# y := str
