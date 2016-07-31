import unittest

class Gotoprogs(unittest.TestCase):
    def setUp(self):
        import esbmc
        # cwd = regression/python
        self.ns, self.opts, self.funcs = esbmc.init_esbmc_process(['test_data/00_big_endian_01/main.c', '--big-endian', '--bv'])

    def get_main(self):
        import esbmc
        return self.funcs.function_map[esbmc.irep_idt('c::main')]

    def test_setup(self):
        self.assertTrue(self.ns != None, "No namespace object generated")
        self.assertTrue(self.opts != None, "No options object generated")
        self.assertTrue(self.funcs != None, "No funcs object generated")

    def test_func_list(self):
        # Should contain the main function and a variety of clib stuff
        funcnames = [x.key().as_string() for x in self.funcs.function_map]
        refnames = ['main', 'c::__ESBMC_assume', 'c::assert', 'c::__ESBMC_assert', 'c::main']
        for name in refnames:
            self.assertTrue(name in funcnames, "func '{}' should be in function map".format(name))

    def test_func_data(self):
        import esbmc
        funcs = [x.data() for x in self.funcs.function_map]
        for x in funcs:
            self.assertTrue(type(x) == esbmc.goto_programs.goto_functiont, "Non-function in function map")

    def test_func_type(self):
        print self.get_main().type
