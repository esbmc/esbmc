#!/usr/bin/python

import esbmc

class Z3python(esbmc.solve.smt_convt):
  def __init__(self, ns):
    super(Z3python, self).__init__(False, ns, False)

  def mk_func_app(self, sort, k, args):
    assert False

  def assert_ast(self, ast):
    assert False

  def dec_solve(self):
    assert False

  def solve_text(self):
    assert False

  def l_get(self, ast):
    assert False

  def mk_sort(self, args):
    assert False

  def mk_smt_int(self, theint, sign):
    assert False

  def mk_smt_bool(self, value):
    assert False

  def mk_smt_symbol(self, name, sort):
    assert False

  def mk_smt_real(self, str):
    assert False

  def mk_smt_bvint(self, theint, sign, w):
    assert False

  def get_bool(self, ast):
    assert False

  def get_bv(self, thetype, ast):
    assert False

  def mk_extract(self, a, high, low, s):
    assert False

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
issat = btor.dec_solve()

# This test case should have a counterexample
assert (issat == esbmc.solve.smt_result.sat)

print "succeeded?"
