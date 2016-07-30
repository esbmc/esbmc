import unittest

class Gotoprogs(unittest.TestCase):
    def setUp(self):
        import esbmc
        # cwd = regression/python
        self.ns, self.opts, self.funcs = esbmc.init_esbmc_process(['test_data/00_big_endian_01/main.c', '--big-endian', '--bv'])

    def test_setup(self):
        self.assertTrue(self.ns != None, "No namespace object generated")
        self.assertTrue(self.opts != None, "No options object generated")
        self.assertTrue(self.funcs != None, "No funcs object generated")
