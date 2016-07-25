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
