#!/usr/bin/python

# PYTHONPATH needs to include the path to $Z3DIR/python.
# And on debian jessie for some reason I need to LD_PRELOAD librt.so?

# Future work: facilities for using the ESBMC flatteners to handle subsets of
# SMT. These are available to C++, but currently too sketchy for python right
# now.

# XXX -- can't print sort.sort because it has a gratuitous const in it?

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
        self.conv = convobj

    def ite(self, conv, cond, falseop):
        assert False

    def eq(self, conv, other):
        new_ast_ref = self.ast == other.ast
        new_ast = Z3ast(new_ast_ref, self.conv, self.conv.bool_sort)
        # Also manually stash this ast
        self.conv.ast_list.append(new_ast)
        return new_ast

    def assign(self, conv, sym):
        assert False

    def update(self, conv, value, idx, idx_expr):
        # Either a tuple update or an array update. Alas, all the exprs baked
        # into ESBMC make no distinguishment.
        if self.sort.id == esbmc.type.type_ids.array:
            res = z3.Update(self.ast, idx.ast, value.ast)
        else:
            assert self.sort.id == esbmc.type.type_ids.struct
            assert False
        result = Z3ast(res, self.conv, self.sort)
        # Also manually stash this ast
        self.conv.ast_list.append(new_ast)
        return result

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
        self.solver = z3.Solver(solver=None, ctx=self.ctx)

        self.func_map = {
            esbmc.solve.smt_func_kind.eq :
                lambda self, args: args[0] == args[1]
        }

        # Various accounting structures for the address space modeling need to
        # be set up, but that needs to happen after the solver is online. Thus,
        # we have to call this once the object is ready to create asts.
        self.smt_post_init()

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

    @stash_ast
    def mk_func_app(self, sort, k, args):
        if k in self.func_map:
            z3ast = self.func_map[k](self, args)
            return Z3ast(z3ast, self, sort)
        print "Unimplemented SMT function {}".format(k)
        assert False

    def assert_ast(self, ast):
        self.solver.add(ast.ast)

    def dec_solve(self):
        assert False

    def solve_text(self):
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
            result.dom_sort = blah
            result.range_sort = blah
        else:
            print kind
            assert False

    @stash_sort
    def mk_struct_sort(self, t):
        # Due to the sins of the fathers, struct arrays get passed through this
        # api too. Essentially, it's ``anything that contains a struct''.
        if t.type_id == esbmc.type.type_ids.struct:
            return self.mk_struct_sort2(t)
        else:
            subtype = esbmc.downcast_type(t.subtype)
            struct_sort = self.mk_struct_sort2(subtype)
            dom_width = self.calculate_array_domain_width(t)
            width_sort = z3.BitVecSort(dom_width, self.ctx)
            arr_sort = z3.ArraySort(width_sort, struct_sort.sort)
            result = Z3sort(arr_sort, esbmc.solve.smt_sort_kind.array, 1, dom_width)
            result.dom_sort = Z3sort(width_sort, esbmc.solve.smt_sort_kind.bv, dom_width)
            result.range_sort = struct_sort
            return result

    def mk_struct_sort2(self, t):
        # Z3 tuples don't _appear_ to be exported to python. Therefore we have
        # to funnel some pointers into it manually, via ctypes.
        num_fields = len(t.member_names)

        # Create names for fields. The multiplication syntax is ctypes way to
        # allocate an array.
        ast_vec = (z3.Symbol * num_fields)()
        i = 0
        for x in t.member_names:
            ast_vec[i] = z3.Symbol(x.as_string())
            i += 1

        # Create types. These are Z3sorts, that contain a z3.SortRef, which
        # in turn contains a z3.Sort. The latter is what we need to funnel
        # into ctype function call.
        sort_vec = (z3.Sort * num_fields)()
        i = 0
        for x in t.members:
            s = self.convert_sort(x)
            sort_vec[i] = s.sort.ast
            i += 1

        # Name for this type
        z3_sym = z3.Symbol(t.typename.as_string())

        # Allocate output ptrs -- function for creating the object, and for
        # projecting fields.
        ret_decl = (z3.FuncDecl * 1)()
        proj_decl = (z3.FuncDecl * 1)()
        sort_ref = z3.Z3_mk_tuple_sort(self.ctx.ctx, z3_sym, num_fields, ast_vec, sort_vec, ret_decl, proj_decl)

        # Reference management: output operands start with zero references IIRC,
        # We want to keep a handle on the returned sort_ref, and the FuncDecl
        # typed ast, for creation of new tuples. The projection decls need to
        # be kept so that we can extract fields from the tuple.
        finsort = Z3sort(z3.BoolSortRef(sort_ref, self.ctx), esbmc.solve.smt_sort_kind.struct)
        proj_decls = [z3.FuncDeclRef(x) for x in proj_decl]
        finsort.decl_ref = z3.FuncDeclRef(ret_decl[0])
        finsort.proj_decls = proj_decls
        return finsort

    @stash_ast
    def tuple_create(self, expr):
        # This is another facility we have to implement with z3's ctypes
        # interface.
        # First convert all expr fields to being z3 asts
        asts = [self.convert_ast(x) for x in expr.members]
        ast_array = (z3.Ast * len(asts))()
        for x in range(len(asts)):
            ast_array[x] = asts[x].ast.ast #really

        # Create the corresponding type
        tsort = self.convert_sort(expr.type)

        tast = z3.Z3_mk_app(self.ctx.ctx, tsort.decl_ref.ast, len(asts), ast_array)
        tref = z3.ExprRef(tast)
        return Z3ast(tref, self, tsort)

    def mk_smt_int(self, theint, sign):
        assert False

    @stash_ast
    def mk_smt_bool(self, value):
        return Z3ast(z3.BoolVal(value, self.ctx), self, self.bool_sort)

    @stash_ast
    def mk_smt_symbol(self, name, sort):
        z3var = z3.Const(name, sort.sort)
        return Z3ast(z3var, self, sort)

    @stash_ast
    def mk_tuple_symbol(self, name, sort):
        # In z3, tuple symbols are the same as normal symbols
        z3var = z3.Const(name, sort.sort)
        return Z3ast(z3var, self, sort)

    @stash_ast
    def mk_tuple_array_symbol(self, expr):
        # Same for tuple arrays
        assert type(expr) == esbmc.expr.symbol
        sort = self.convert_sort(expr.type)
        z3var = z3.Const(expr.name.as_string(), sort.sort)
        return Z3ast(z3var, self, sort)

    def mk_smt_real(self, str):
        assert False

    @stash_ast
    def mk_smt_bvint(self, theint, sign, w):
        bvsort = self.mk_sort([esbmc.solve.smt_sort_kind.bv, w])
        z3ast = z3.BitVecVal(theint.to_long(), w, self.ctx)
        return Z3ast(z3ast, self, bvsort)

    def get_bool(self, ast):
        assert False

    def get_bv(self, thetype, ast):
        assert False

    def l_get(self, ast):
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
