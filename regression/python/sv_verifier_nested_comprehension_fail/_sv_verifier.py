# This file is part of the SV-Benchmarks collection of verification tasks:
# https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
#
# SPDX-FileCopyrightText: 2026 Raphaël Monat, Inria
# SPDX-FileCopyrightText: 2026-... The SV-Benchmarks Community
#
# SPDX-License-Identifier: Apache-2.0

import random
import string
from typing import Callable

def nondet_bool():
    return bool(random.randint(0, 2))

def nondet_int() -> int:
    # unbounded int
    i = 0
    while nondet_bool():
        i += 1
    if nondet_bool():
        i = -i
    return i

def nondet_list(nondet_elem):
    l = []
    while nondet_bool():
        l.append(nondet_elem())
    return l

def nondet_dict(nondet_key, nondet_value):
    d = dict()
    while nondet_bool():
        d[nondet_key()] = nondet_value()
    return d

def nondet_str():
    # Source - https://stackoverflow.com/a/23728630
    # Posted by Randy Marsh, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-06-01, License - CC BY-SA 4.0
    # Note that this is the concrete implementation, but it could be changed to include *any* kind of caracter
    return ''.join(random.choice(string.printable) for _ in range(abs(nondet_int())))

def nondet_float():
    import sys
    if nondet_bool():
        return random.choice([float('inf'), float('-inf'), float('nan')])
    return random.uniform(sys.float_info.min, sys.float_info.max)
