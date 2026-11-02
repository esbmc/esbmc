# This file is part of the SV-Benchmarks collection of verification tasks:
# https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
#
# SPDX-FileCopyrightText: 2026 Mostafa Hassan, Simon Wehrli, Caterina Urban, Marco Eilers
#
# SPDX-License-Identifier: MPL-2.0

import _sv_verifier

class A:
    def __init__(self, x):
        self.x = 1

    def __gt__(self, other):
        return self.x > other.x

    def __lt__(self, other):
        return self.x < other.x

    def __ge__(self, other):
        return self.x >= other.x

    def __le__(self, other):
        return self.x <= other.x

a1 = A(1)
a2 = A(2)
b3 = a1 > a2
b4 = a1 >= a2
b5 = a1 < a2
b6 = a1 <= a2
b7 = "" > ""
b8 = 1 > 1
b9 = [1,2,3] > [4,6]
b10 = {1} >= {3}
_sv_verifier.check_type(a1, A)
_sv_verifier.check_type(a2, A)
_sv_verifier.check_type(a1.x, int)
_sv_verifier.check_type(a2.x, int)
_sv_verifier.check_type(b3, int)
_sv_verifier.check_type(b4, int)
_sv_verifier.check_type(b5, int)
_sv_verifier.check_type(b6, int)
_sv_verifier.check_type(b7, int)
_sv_verifier.check_type(b8, int)
_sv_verifier.check_type(b9, int)
_sv_verifier.check_type(b10, int)
