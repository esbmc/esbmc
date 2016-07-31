import unittest

class Gotoinsns(unittest.TestCase):
    def setUp(self):
        import esbmc
        # cwd = regression/python
        self.ns, self.opts, self.funcs = esbmc.init_esbmc_process(['test_data/00_big_endian_01/main.c', '--big-endian', '--bv'])
        self.main = self.funcs.function_map[esbmc.irep_idt('c::main')].body
        self.insns = self.main.get_instructions()

    def tearDown(self):
        import esbmc
        esbmc.kill_esbmc_process()
        self.ns, self.opts, self.funcs = None, None, None
        self.main = None

    def test_basic(self):
        import esbmc
        self.assertTrue(type(self.insns[0]) == esbmc.goto_programs.instructiont, "Insns should have insn type")

    def test_target(self):
        # At least one insn should have a non-nil target. For our example.
        targets = [x.target for x in self.insns if x.target != None]
        self.assertTrue(len(targets) != 0, "At least one insn in main should have branch")

    def test_first_insn(self):
        import esbmc
        theinsn = self.insns[0]
        self.assertTrue(theinsn.type == esbmc.goto_programs.goto_program_instruction_type.OTHER, "Wrong insn type")
        code = esbmc.downcast_expr(theinsn.code)
        self.assertTrue(code.expr_id == esbmc.expr.expr_ids.code_decl, "decl insn has wrong expr type")
        self.assertTrue(code.value.as_string() == "c::main::main::1::i", "decl insn has wrong expr value")
        self.assertTrue(theinsn.function.as_string() == "c::main", "decl insn has wrong function name")
