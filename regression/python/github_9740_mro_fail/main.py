# This file is part of the SV-Benchmarks collection of verification tasks:
# https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
#
# SPDX-FileCopyrightText: 2026 Mostafa Hassan, Simon Wehrli, Caterina Urban, Marco Eilers
#
# SPDX-License-Identifier: MPL-2.0

import _sv_verifier

class A:
    def who_am_i(self):
        return A()
class B(A):
    pass
class C(A):
    def who_am_i(self):
        return C()
class D(B, C):
    pass
d1 = D()
who = d1.who_am_i()
_sv_verifier.check_type(d1, D)
_sv_verifier.check_type(who, B)
"""With the old MRO algorithm, the type of `who` would be Type[A], because
The naive DFS will reach A before C
With the C3 Linearization algorithm, C will appear first in the search context
so the type of `who` is Type[C]
"""
