import unittest

class Gotofuncs(unittest.TestCase):
    def setUp(self):
        import esbmc
        # cwd = regression/python
        self.ns, self.opts, self.po = esbmc.init_esbmc_process(['test_data/00_big_endian_01/main.c', '--big-endian', '--bv'])
        self.funcs = self.po.goto_functions

    def tearDown(self):
        import esbmc
        esbmc.kill_esbmc_process()
        self.ns, self.opts, self.funcs = None, None, None

    def get_main(self):
        import esbmc
        return self.funcs.function_map[esbmc.irep_idt('main')]

    def test_setup(self):
        self.assertTrue(self.ns != None, "No namespace object generated")
        self.assertTrue(self.opts != None, "No options object generated")
        self.assertTrue(self.funcs != None, "No funcs object generated")

    def test_func_list(self):
        # Should contain the main function and a variety of clib stuff
        funcnames = [x.key().as_string() for x in self.funcs.function_map]
        refnames = ['__ESBMC_main', 'main']
        for name in refnames:
            self.assertTrue(name in funcnames, "func '{}' should be in function map".format(name))

    def test_func_data(self):
        import esbmc
        funcs = [x.data() for x in self.funcs.function_map]
        for x in funcs:
            self.assertTrue(type(x) == esbmc.goto_programs.goto_functiont, "Non-function in function map")

    def test_func_type(self):
        import esbmc
        # Access type field
        thetype = self.get_main().type
        # The function type is of code type, i.e. registers the args
        # and the return type
        self.assertTrue(type(thetype) == esbmc.type.code_typet, "goto prog field's type should be old code_typet")
        # Can't explicitly construct a type2tc from this, but we can
        # pass it through the downconverter which will achieve the same thing
        newtype = esbmc.downcast_type(self.get_main().type)
        self.assertTrue(newtype.type_id == esbmc.type.type_ids.code, "Function type should have code type")
        # Other elements of that code type are unrelated to the matter of
        # testing this module

    def test_func_body_present(self):
        main = self.get_main()
        self.assertTrue(main.body_available, "Body of main should be present")

    def test_add_function(self):
        import esbmc
        main = self.get_main()
        self.funcs.function_map[esbmc.irep_idt('coveredinbees_add_funcion_test')] = main
        # Correctness of this is that it doesn't throw

    def test_update_funcs(self):
        main = self.get_main()
        insns = main.body.get_instructions()
        self.assertTrue(insns[1].location_number != 0, "Second insn of main should not be zero")
        insns[1].location_number = 0
        self.assertTrue(insns[1].location_number == 0, "Second insn of main should have been set to zero")
        main.body.set_instructions(insns)

        self.funcs.update()

        # That should have reset the location number
        insns = main.body.get_instructions()
        self.assertTrue(insns[1].location_number != 0, "Second insn of main should not be zero after update")
