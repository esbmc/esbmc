#!/usr/bin/python
import esbmc
from z3_solver import Z3python

# Cruft for running below

ns, opts, po = esbmc.init_esbmc_process(['../regression/python/test_data/00_big_endian_01/main.c', '--big-endian', '--bv'])
funcs = po.goto_functions
main = funcs.function_map[esbmc.irep_idt('c::main')].body
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
