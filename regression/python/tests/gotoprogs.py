import unittest

class Gotoprogs(unittest.TestCase):
    def setUp(self):
        import esbmc
        # cwd = regression/python
        self.ns, self.opts, self.funcs = esbmc.init_esbmc_process(['test_data/00_big_endian_01/main.c', '--big-endian', '--bv'])
        self.main = self.funcs.function_map[esbmc.irep_idt('c::main')].body

    def tearDown(self):
        import esbmc
        esbmc.kill_esbmc_process()
        self.ns, self.opts, self.funcs = None, None, None
        self.main = None

    def test_get_insns(self):
        insns = self.main.get_instructions()
        self.assertTrue(insns != None, "Couldn't get instructions")

    def test_set_insns(self):
        insns = self.main.get_instructions()
        self.main.set_instructions(insns)
        # Correctness is that this doesn't throw.

    def test_print(self):
        string = self.main.to_string()
        self.assertTrue("i=258f;" in string, "Main function printed string should contain certain constant")
