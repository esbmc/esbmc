# This file is part of the SV-Benchmarks collection of verification tasks:
# https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
#
# SPDX-FileCopyrightText: 2026 Mostafa Hassan, Simon Wehrli, Caterina Urban, Marco Eilers
#
# SPDX-License-Identifier: MPL-2.0

import _sv_verifier
# class_type_params {'Concatenator': [0]}
# type_params {'concatenate': [1,2]}

class Concatenator:
    def __init__(self, a):
        self.a = a
    def concatenate(self, separator, from_, to):
        self.a = separator
        to.append(separator)
        for f in from_:
            to.append(f)
class Z:
    def foo(self):
        pass
class Y(Z):
    def bar(self):
        pass
class X(Y):
    def baz(self):
        pass
class W(X):
    def omg(self):
        pass
concatenator = Concatenator(Y())
concatenator.a.bar()
_sv_verifier.check_type(concatenator, Concatenator)
wlst = [W()]
wlst[0].omg()
xlst = [X()]
xlst[0].baz()
concatenator.concatenate(X(), wlst, xlst)
_sv_verifier.check_type(wlst, list)
_sv_verifier.check_type(xlst, list)
_sv_verifier.check_type(wlst[0], W)
_sv_verifier.check_type(xlst[0], X)
class ZP:
    def foo(self):
        pass
class YP(ZP):
    def bar(self):
        pass
class XP(YP):
    def baz(self):
        pass
class WP(XP):
    def omg(self):
        pass
concatenatorP = Concatenator(YP())
concatenatorP.a.bar()
_sv_verifier.check_type(concatenatorP, Concatenator)
wlstP = [WP()]
wlstP[0].omg()
xlstP = [XP()]
xlstP[0].baz()
concatenatorP.concatenate(XP(), wlstP, xlstP)
_sv_verifier.check_type(wlstP, list)
_sv_verifier.check_type(xlstP, list)
_sv_verifier.check_type(wlstP[0], WP)
_sv_verifier.check_type(xlstP[0], XP)
