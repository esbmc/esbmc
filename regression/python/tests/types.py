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
