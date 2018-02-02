import unittest

class Exprs(unittest.TestCase):
    def make_unsigned(self, width=32):
        import esbmc
        return esbmc.type.unsignedbv.make(width)

    def make_bigint(self, value=0):
        import esbmc
        return esbmc.BigInt(value)

    def make_int(self, value=0, width=32):
        import esbmc
        u = self.make_unsigned(width)
        val = self.make_bigint(value)
        return esbmc.expr.constant_int.make(u, val)

    def test_min(self):
        import esbmc
        u32 = self.make_unsigned()
        zero = self.make_bigint()
        constint = esbmc.expr.constant_int.make(u32, zero)

        self.assertTrue(constint != None, "Can't create constant_int2t")

    def test_fields(self):
        import esbmc
        val = self.make_int()
        self.assertTrue(val.type == self.make_unsigned(), "Can't get expr type field")
        self.assertTrue(val.value == self.make_bigint(), "Can't get expr value field")
        self.assertTrue(val.expr_id == esbmc.expr.expr_ids.constant_int, "Can't get expr value field")

    def test_cmps(self):
        import esbmc
        val0 = self.make_int(0)
        val1 = self.make_int(1)
        self.assertTrue(val0 == val0, "Same comparison should be true")
        self.assertFalse(val0 != val0, "Same comparison should still be true")
        self.assertTrue(val0 != val1, "Different comparision should be unequal")
        self.assertFalse(val0 == val1, "Different comparision shouldn't be equal")
        self.assertTrue(val0 < val1, "constints should compare less than")
        self.assertFalse(val1 < val0, "reversed constints should not compare less than")

    def test_pretty(self):
        import esbmc
        val = self.make_int()
        reftext = "constant_int\n* value : 0\n* type : unsignedbv\n  * width : 32"
        self.assertTrue(val.pretty(0) == reftext, "Expr pretty should work")

    def test_clone(self):
        import esbmc
        val = self.make_int()
        val1 = val.clone()
        self.assertTrue(val == val1, "Cloned expr should be identical")
        self.assertFalse(val is val1, "Cloned expr should not be same object")

    def test_add(self):
        import esbmc
        val = self.make_int(1)
        val1 = self.make_int(2)
        add = esbmc.expr.add.make(val.type, val, val1)
        self.assertTrue(add.expr_id == esbmc.expr.expr_ids.add, "Add expr should have add id")
        self.assertTrue(add.side_1 == val, "Incorrect add2t field")
        self.assertTrue(add.side_2 == val1, "Incorrect add2t field")
        self.assertFalse(add.side_1 == val1, "add2t field shouldn't compare true")
        self.assertFalse(add.side_2 == val, "add2t field shouldn't compare true")
        # Accessing struct field should different python objects
        self.assertFalse(add.side_1 is val, "Incorrect python object comparison")
        self.assertFalse(add.side_2 is val1, "Incorrect python object comparison")

    def test_downcast(self):
        import esbmc
        val = self.make_int(1)
        val1 = self.make_int(2)
        add = esbmc.expr.add.make(val.type, val, val1)
        # Fields should all have type expr2t
        self.assertTrue(type(add.side_1) == esbmc.expr.expr2t, "Wrong field type in expr")
        downcasted = esbmc.downcast_expr(add.side_1)
        self.assertTrue(type(downcasted) == esbmc.expr.constant_int, "Downcast failed")
        # Should be brought into python as a different object
        self.assertFalse(downcasted is val, "downcasted object should be new pyref")

    def test_downcast_none(self):
        import esbmc
        downcasted = esbmc.downcast_expr(None)
        self.assertTrue(downcasted == None, "Downcast of none should be none")

    def test_none(self):
        import esbmc
        val = self.make_int()
        # Technically an illegal irep, might break if we get more stringent
        add = esbmc.expr.add.make(val.type, val, None)
        self.assertTrue(val != None, "cmp with none failed")
        self.assertTrue(add.side_2 == None, "cmp with referenced none failed")
        self.assertTrue(esbmc.expr.is_nil_expr(add.side_2), "is-nil comparison failed")

    def test_except(self):
        import esbmc
        u32 = self.make_unsigned()
        try:
            arr = esbmc.expr.add.make(u32, u32, u32)
        except TypeError:
            pass
        else:
            self.assertTrue(False, "add irep construction should have thrown")

    def test_bools(self):
        import esbmc
        letrue = esbmc.expr.constant_bool.make(True)
        self.assertTrue(letrue.value, "constant bool has wrong value")
        self.assertTrue(type(letrue.value) is bool, "constant bool has wrong type")

    def test_sideeffect(self):
        # Test one of the rarer enums
        import esbmc
        val = self.make_int()
        val1 = self.make_int(1)
        se = esbmc.expr.sideeffect.make(val.type, val, val1, esbmc.expr.expr_vec(), val.type, esbmc.expr.sideeffect_allockind.malloc)
        self.assertTrue(se != None, "Couldn't create side-effect")
        self.assertTrue(se.type == val.type, "Sideeffect has wrong type")
        self.assertTrue(se.operand == val, "Sideeffect has wrong operand")
        self.assertTrue(se.size == val1, "Sideeffect has wrong size value")
        self.assertTrue(len(se.arguments) == 0, "Sideeffect has argument values")
        self.assertTrue(se.alloctype == val.type, "Sideeffect allockind has wrong value")
        self.assertTrue(se.kind == esbmc.expr.sideeffect_allockind.malloc, "Sideeffect should have malloc kind")

    def test_symbol(self):
        import esbmc
        u32 = self.make_unsigned()
        name = esbmc.irep_idt("fgasdf")
        sym = esbmc.expr.symbol.make(u32, name, esbmc.expr.symbol_renaming.level2, 1, 2, 3, 4)
        self.assertTrue(sym.type == u32, "Symbol has wrong type")
        self.assertTrue(sym.name == name, "Symbol has wrong name")
        self.assertTrue(sym.renamelev == esbmc.expr.symbol_renaming.level2, "Symbol has wrong renaming level")
        self.assertTrue(sym.level1_num == 1, "Symbol has wrong level1 num")
        self.assertTrue(sym.level2_num == 2, "Symbol has wrong level2 num")
        self.assertTrue(sym.thread_num == 3, "Symbol has wrong thread num")
        self.assertTrue(sym.node_num == 4, "Symbol has wrong node num")

    def test_crc(self):
        import esbmc
        val = self.make_int()
        # In rare occasions if alg changes, might actually be 0.
        self.assertTrue(val.crc() != 0, "expr crc failed")

    def test_depth(self):
        import esbmc
        val = self.make_int(1)
        val1 = self.make_int(2)
        add = esbmc.expr.add.make(val.type, val, val1)
        self.assertTrue(add.depth() == 2, "Incorrect irep depth")

    def test_num_nodes(self):
        import esbmc
        val = self.make_int(1)
        val1 = self.make_int(2)
        add = esbmc.expr.add.make(val.type, val, val1)
        self.assertTrue(add.num_nodes() == 3, "Incorrect irep node num")

    def test_simplify(self):
        import esbmc
        val = self.make_int(1)
        val1 = self.make_int(2)
        add = esbmc.expr.add.make(val.type, val, val1)
        simp = add.simplify()
        self.assertTrue(simp != None, "Could not simplify an add")
        self.assertTrue(simp.expr_id == esbmc.expr.expr_ids.constant_int, "Simplified add should be a constant int")
        # As it's a new irep it's copied by container value, so gets
        # downcasted automagically by boost.python?
        self.assertTrue(simp.value.to_long() == 3, "Simplified add has wrong value")

    def test_struct(self):
        import esbmc
        from .types import Types
        # Type where we have four fields (a, b, c, d) of integer sizes one
        # to four.
        struct_type = Types.struct_maker()

        # Create integers that'll fit into those types
        values = [(0, 1), (1, 2), (2, 3), (3, 4)]
        vec_of_ints = [self.make_int(x, y) for x, y in values]
        # Pump those into an expr vector
        expr_vec = esbmc.expr.expr_vec()
        self.assertTrue(expr_vec != None, "Couldn't create expr_vec")
        expr_vec.extend(vec_of_ints)

        # Aaannnddd, create a struct
        struct = esbmc.expr.constant_struct.make(struct_type, expr_vec)
        self.assertTrue(struct != None, "Couldn't create constant struct")
        i = 0
        for x in struct.members:
            self.assertTrue(x == vec_of_ints[i], "Struct contents mistmatch")
            i = i + 1
        self.assertTrue(struct.type == struct_type, "Struct type mismatch")

    def test_fixedbv(self):
        import esbmc
        fbv = esbmc.fixedbv()
        fbv_spec = esbmc.fixedbv_spec(32, 32)
        fbv.spec = fbv_spec
        fbv.from_integer(esbmc.BigInt(0))

        fbvt = esbmc.type.fixedbv.make(32, 32)
        const_fbv = esbmc.expr.constant_fixedbv.make(fbv)
        reftext = "constant_fixedbv\n* value : 0\n* type : fixedbv\n  * width : 32\n  * integer_bits : 32"
        self.assertTrue(const_fbv.pretty(0) == reftext, "Created fixedbv has wrong form")

    def test_sym_compare(self):
        import esbmc
        # Build an expr we indirectly access and a symbol
        foo = self.make_int()
        ubv = self.make_unsigned()
        add = esbmc.expr.add.make(ubv, foo, foo)
        idt = esbmc.irep_idt("fgasfd")
        lev = esbmc.expr.symbol_renaming.level0
        sym = esbmc.expr.symbol.make(ubv, idt, lev, 0, 0, 0, 0)

        # Problem: when we downcast this b.p knows that 'sym' is a symbol2tc.
        # But it doesn't know for some reason that it can just cast that to
        # a expr2tc, via inheritance. Test for this -- __eq__ returns
        # NotImplemented when the operator== call construction fails.
        sym = esbmc.downcast_expr(sym)
        foo2 = add.side_1
        self.assertFalse(foo2.__eq__(sym) == NotImplemented, "Downcasted expr should be castable to expr2tc")

    def test_call_none(self):
        import esbmc
        # Create what is, admitedly, an invalid irep
        add = esbmc.expr.add.make(None, None, None)
        try:
            # This should fail lvalue conversion: can't make an lvalue out of NULL
            add.side_1.pretty(0)
        except TypeError:
            pass
        else:
            self.assertTrue(False, "Null-to-expr conversion should have failed")

    def test_iter(self):
        import esbmc
        # Build a nontrivial expr
        foo = self.make_int()
        ubv = self.make_unsigned()
        idt = esbmc.irep_idt("fgasfd")
        lev = esbmc.expr.symbol_renaming.level0
        sym = esbmc.expr.symbol.make(ubv, idt, lev, 0, 0, 0, 0)
        add = esbmc.expr.add.make(ubv, foo, sym)

        it = iter(add)
        obj1 = next(it)
        self.assertTrue(obj1 == foo, "Object iterator obj1 not as expected")
        obj2 = next(it)
        self.assertTrue(obj2 == sym, "Object iterator obj2 not as expected")
        try:
            next(it)
            self.assertTrue(False, "Object iterator should not have completed 3rd time")
        except StopIteration:
            pass

    # When we iterate over an expression, we should detach from the original
    # expressions to avoid unexpected mutations.
    def test_iter_detach(self):
        import esbmc
        # Build a nontrivial expr
        foo = self.make_int()
        ubv = self.make_unsigned()
        idt = esbmc.irep_idt("fgasfd")
        lev = esbmc.expr.symbol_renaming.level0
        sym = esbmc.expr.symbol.make(ubv, idt, lev, 0, 0, 0, 0)
        add = esbmc.expr.add.make(ubv, foo, sym)

        addclone = add.clone()
        it = iter(addclone)
        obj1 = next(it)
        self.assertTrue(obj1 == foo, "Object iterator obj1 not as expected")
        obj2 = next(it)
        self.assertTrue(obj2 == sym, "Object iterator obj2 not as expected")

        try:
            next(it)
            self.assertTrue(False, "Object iterator should not have completed 3rd time")
        except StopIteration:
            pass

        obj2 = esbmc.downcast_expr(obj2)
        self.assertTrue(esbmc.expr.expr_ids.symbol == obj2.expr_id, "Downcasted symbol should be sym id'd")
        obj2.iknowwhatimdoing_level2_num = 1
        self.assertTrue(obj2.level2_num != sym.level2_num, "Iterated object should have detached from original")

    def test_in(self):
        import esbmc
        # Build a nontrivial expr
        foo = self.make_int()
        ubv = self.make_unsigned()
        idt = esbmc.irep_idt("fgasfd")
        lev = esbmc.expr.symbol_renaming.level0
        sym = esbmc.expr.symbol.make(ubv, idt, lev, 0, 0, 0, 0)
        add = esbmc.expr.add.make(ubv, foo, sym)

        nonzero = self.make_int(value=1)

        # Assert that the sub expressions are "in" the composite
        self.assertTrue(foo in add, "Couldn't find subexpression 1 in expr")
        self.assertTrue(sym in add, "Couldn't find subexpression 1 in expr")
        self.assertFalse(nonzero in add, "Separate expr shouldn't be in main expr")
        self.assertTrue(add in add, "Couldn't find expr in itself")
