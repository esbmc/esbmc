#!/usr/bin/python
import sys
import esbmc
from z3_solver import Z3python

# Cruft for running below. Pass command line options through to esbmc.
# Hopefully this means we effectively wrap esbmc.

ns, opts, po = esbmc.init_esbmc_process(sys.argv[1:])
eq = esbmc.symex.equation(ns)
art = esbmc.symex.reachability_tree(po.goto_functions, ns, opts, eq, po.context, po.message_handler)

art.setup_for_new_explore()
result = art.get_next_formula()
lolsolve = Z3python(ns)
result.target.convert(lolsolve)
issat = lolsolve.dec_solve()

# This test case should have a counterexample
assert (issat == esbmc.solve.smt_result.sat)

print "succeeded?"
