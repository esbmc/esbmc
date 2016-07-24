import unittest

class Types(unittest.TestCase):
    def setUp(self):
        from esbmc import BigInt
        self.thirtytwo = BigInt(32)

    def make_u32(self):
        import esbmc
        return esbmc.type.unsignedbv.make(32)

    def test_empty(self):
        import esbmc
        emptytype = esbmc.type.empty.make()
        self.assertTrue(emptytype != None, "Can't create empty type")

    def test_bv(self):
        import esbmc
        unsignedbv = self.make_u32()
        self.assertTrue(unsignedbv != None, "Can't create bv type")

    def test_bv_data(self):
        import esbmc
        unsignedbv = self.make_u32()
        self.assertTrue(unsignedbv.width == 32, "Can't access bv type field")

    def test_bv_typeid(self):
        import esbmc
        unsignedbv = self.make_u32()
        self.assertTrue(unsignedbv.type_id == esbmc.type.type_ids.unsignedbv,
                "Can't access bv type_id field")

    def test_bv_pretty(self):
        import esbmc
        unsignedbv = self.make_u32()
        reftext = "unsignedbv\n* width : 32"
        self.assertTrue(unsignedbv.pretty(0) == reftext,
                "Can't pretty print types")
