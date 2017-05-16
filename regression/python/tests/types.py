import unittest

class Types(unittest.TestCase):
    def setUp(self):
        from esbmc import BigInt
        self.thirtytwo = BigInt(32)

    def make_unsigned(self, width=32):
        import esbmc
        return esbmc.type.unsignedbv.make(width)

    def test_empty(self):
        import esbmc
        emptytype = esbmc.type.empty.make()
        self.assertTrue(emptytype != None, "Can't create empty type")

    def test_bv(self):
        import esbmc
        unsignedbv = self.make_unsigned()
        self.assertTrue(unsignedbv != None, "Can't create bv type")

    def test_bv_data(self):
        import esbmc
        unsignedbv = self.make_unsigned()
        self.assertTrue(unsignedbv.width == 32, "Can't access bv type field")

    def test_bv_typeid(self):
        import esbmc
        unsignedbv = self.make_unsigned()
        self.assertTrue(unsignedbv.type_id == esbmc.type.type_ids.unsignedbv,
                "Can't access bv type_id field")

    def test_bv_pretty(self):
        import esbmc
        unsignedbv = self.make_unsigned()
        reftext = "unsignedbv\n* width : 32"
        self.assertTrue(unsignedbv.pretty(0) == reftext,
                "Can't pretty print types")

    def test_bv_equality(self):
        import esbmc
        u32 = self.make_unsigned(32)
        u16 = self.make_unsigned(16)
        self.assertTrue(u32 == u32, "Type comparison fails")
        self.assertTrue(u32 != u16, "Type comparison fails")

    def test_bv_clone(self):
        import esbmc
        u32 = self.make_unsigned()
        u32_c = u32.clone()
        self.assertTrue(u32 == u32_c, "Couldn't clone unsigned32")
        self.assertFalse(u32 is u32_c, "Clone returned same reference")

    def test_nested(self):
        import esbmc
        u32 = self.make_unsigned()
        ptr = esbmc.type.pointer.make(u32)
        self.assertTrue(ptr != None, "Couldn't create ptr type")
        self.assertTrue(ptr.subtype == u32, "Ptr type subtype wrong")
        self.assertFalse(ptr.subtype is u32, "Ptr type subtype wrong")

    def test_downcast(self):
        import esbmc
        u32 = self.make_unsigned()
        ptr = esbmc.type.pointer.make(u32)
        self.assertTrue(type(ptr.subtype) == esbmc.type.type2t, "Pointer subtype field should be type2t")
        foo = esbmc.downcast_type(ptr.subtype)
        self.assertTrue(type(foo) == esbmc.type.unsignedbv, "Pointer subtype field should _contain_ unsignedbv")

    def test_array(self):
        import esbmc
        u32 = self.make_unsigned()
        arr = esbmc.type.array.make(u32, None, True)
        self.assertTrue(arr != None, "Failed to construct array type")
        self.assertTrue(arr.subtype == u32, "Array subtype has wrong value")
        self.assertTrue(arr.array_size == None, "Array size should be None")
        self.assertTrue(arr.size_is_infinite, "Array inf field has wrong value")

    def test_internal_ref(self):
        import esbmc
        u32 = self.make_unsigned()
        arr = esbmc.type.array.make(u32, None, True)
        # Important: when accessing sub-fields of ireps, boost.python generates
        # internal references to it, which is *not* None, even if the contained
        # thing is an empty irep. But comparisons work.
        self.assertFalse(arr.array_size is None, "irep field should be non-None")
        self.assertTrue(arr.array_size == None, "irep field should _compare_ to be None")
        self.assertTrue(esbmc.expr.is_nil_expr(arr.array_size), "irep field should be nil")

    def test_none(self):
        import esbmc
        self.assertTrue(esbmc.type.is_nil_type(None), "None should convert to type2tc")

    def test_cmp(self):
        import esbmc
        u16 = esbmc.type.unsignedbv.make(16)
        u32 = esbmc.type.unsignedbv.make(32)
        self.assertTrue(u16 != u32, "type inequality should be true")
        self.assertFalse(u16 == u32, "type equality should be false")
        self.assertTrue(u16 < u32, "type less-than should be true")

    def test_type_check(self):
        import esbmc
        u32 = self.make_unsigned()
        arr = esbmc.type.array.make(u32, None, True)
        # Ensure that we can't pass an expr into is_nil_type
        try:
            esbmc.type.is_nil_type(arr.array_size)
        except TypeError:
            self.assertTrue(True)
        else:
            self.assertTrue(False, "Badly typed comparison should have thrown")

    def test_type_vec(self):
        import esbmc
        alist = []
        for x in range(1, 5):
            alist.append(esbmc.type.unsignedbv.make(x))
        tlist = esbmc.type.type_vec()
        tlist.extend(alist)
        self.assertTrue(tlist != None, "should be able to construct type list")

        # Check contents
        for x in range(0, 4):
            self.assertTrue(tlist[x] == alist[x], "type_vec contents differ")

    @staticmethod
    def struct_maker():
        import esbmc
        alist = [esbmc.type.unsignedbv.make(x) for x in range(1, 5)]
        namelist = [esbmc.irep_idt(x) for x in ["a", "b", "c", "d"]]

        tlist = esbmc.type.type_vec()
        tlist.extend(alist)
        ilist = esbmc.irep_idt_vec()
        ilist.extend(namelist)

        # Pretty names = names
        struct = esbmc.type.struct.make(tlist, ilist, ilist, esbmc.irep_idt("fgasdf"))
        return struct

    def test_struct_creation(self):
        # Just ensure there's something in there and it's not nil
        struct = self.struct_maker()
        self.assertTrue(struct.pretty(0) != "", "Structure creation failed")

    def test_struct_contents(self):
        struct = self.struct_maker()
        for x in range(1, 5):
            self.assertTrue(struct.members[x-1] == self.make_unsigned(x), "Structure contents mismatch")

        namelist = ["a", "b", "c", "d"]
        for x in range(0, 4):
            self.assertTrue(struct.member_names[x].as_string() == namelist[x], "Structure contents mismatch")

        self.assertTrue(struct.typename.as_string() == "fgasdf", "Structure contents mismatch")

    def test_call_none(self):
        import esbmc
        ptr = esbmc.type.pointer.make(None)
        try:
            # Should fail lvalue conversion, can't make lvalue out of null
            ptr.subtype.pretty(0)
        except TypeError:
            pass # good
        else:
            self.assertTrue(False, "Lvalue conversion of null type should have thrown")
