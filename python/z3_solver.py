#!/usr/bin/python

# PYTHONPATH needs to include the path to $Z3DIR/python.
# And on debian jessie for some reason I need to LD_PRELOAD librt.so?

# Future work: facilities for using the ESBMC flatteners to handle subsets of
# SMT. These are available to C++, but currently too sketchy for python right
# now.

import esbmc
import z3

class Z3sort(esbmc.solve.smt_sort):
    def __init__(self, z3sort, kind, width=None, domain_width=None):
        if domain_width != None:
            super(Z3sort, self).__init__(kind, width, domain_width)
        elif width != None:
            super(Z3sort, self).__init__(kind, width)
        else:
            super(Z3sort, self).__init__(kind)
        self.sort = z3sort

  # Has no other methods, only provides id / data_width / dom_width in base
  # class, so that the rest of smt_convt can decide the right path.

class Z3ast(esbmc.solve.smt_ast):
    def __init__(self, ast, convobj, sort):
        super(Z3ast, self).__init__(convobj, sort)
        self.ast = ast

    def ite(self, conv, cond, falseop):
        assert False

    def eq(self, conv, other):
        assert False

    def assign(self, conv, sym):
        assert False

    def update(self, conv, sym):
        assert False

    def select(self, conv, idx):
        assert False

    def project(self, conv, elem):
        assert False

class Z3python(esbmc.solve.smt_convt):
    def __init__(self, ns):
        super(Z3python, self).__init__(False, ns, False, True, True)
        self.ctx = z3.Context()
        self.ast_list = []
        self.sort_list = []
        self.bool_sort = self.mk_sort((esbmc.solve.smt_sort_kind.bool,))
        # Various accounting structures for the address space modeling need to
        # be set up, but that needs to happen after the solver is online. Thus,
        # we have to call this once the object is ready to create asts.
        self.init_addr_space_array()

    # Decorator function: return a function that appends the return value to
    # the list of asts. This is vital: the python object returned from various
    # functions needs to live as long as the convt object, so we need to keep
    # a reference to it
    def stash_ast(func):
        def tmp(self, *args, **kwargs):
            ast = func(self, *args, **kwargs)
            self.ast_list.append(ast)
            return ast
        return tmp

    def stash_sort(func):
        def tmp(self, *args, **kwargs):
            sort = func(self, *args, **kwargs)
            self.sort_list.append(sort)
            return sort
        return tmp

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

    @stash_sort
    def mk_sort(self, args):
        kind = args[0]
        if kind == esbmc.solve.smt_sort_kind.bool:
            return Z3sort(z3.BoolSort(self.ctx), kind)
        elif kind == esbmc.solve.smt_sort_kind.bv:
            width = args[1]
            z3sort = z3.BitVecSort(width, self.ctx)
            return Z3sort(z3sort, kind, args[1])
        elif kind == esbmc.solve.smt_sort_kind.array:
            domain = args[1]
            range_ = args[2]
            print domain
            print range_
            assert False
        else:
            print kind
            assert False

    def mk_smt_int(self, theint, sign):
        assert False

    @stash_ast
    def mk_smt_bool(self, value):
        return Z3ast(z3.BoolVal(value, self.ctx), self, self.bool_sort)

    def mk_smt_symbol(self, name, sort):
        print dir(sort)
        print type(sort)
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
