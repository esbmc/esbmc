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

    def test_insn_locations(self):
        import esbmc
        theinsn = self.insns[0]
        loc = esbmc.location.from_locationt(theinsn.location)
        self.assertTrue(loc.file.as_string() == 'test_data/00_big_endian_01/main.c', "File string is wrong")
        self.assertTrue(loc.function.as_string() == "main", "Func name in location is wrong")
        self.assertTrue(loc.line == 5, "Line number in test file is wrong")
        self.assertTrue(loc.column == 0, "Column number in test file is wrong")

    def test_more_fields(self):
        import esbmc
        theinsn = self.insns[0]
        self.assertTrue(esbmc.downcast_expr(theinsn.guard).constant_value == True, "insn guard should be true")
        # No access to labels field right no
        # This has the value 99 right now, but we can't really assert that
        # because it'll change when we edit... anything
        self.assertTrue(theinsn.location_number > 0, "Wrong insn number")
        self.assertTrue(theinsn.loop_number == 0, "Loop number doesn't exist?")
        # This seems to be a useless field
        self.assertTrue(theinsn.target_number == 4294967295, "Target number doesn't exist?")

        norm = theinsn.to_string()
        wloc = theinsn.to_string(True)
        wvars = theinsn.to_string(True, True)
        self.assertTrue(norm != wloc and norm != wvars and wloc != wvars, "var changing flags should change to_string output")

        self.assertTrue(theinsn.is_other(), "is_other method of insn should work")

    def test_insn_clear(self):
        import esbmc
        self.insns[0].clear(esbmc.goto_programs.goto_program_instruction_type.ASSIGN)
        self.assertTrue(self.insns[0].type == esbmc.goto_programs.goto_program_instruction_type.ASSIGN, "Cleared insn should be assign")
        self.assertTrue(self.insns[0].code == None, "Cleared insn should have nil irep")

    def test_insn_printing(self):
        self.assertTrue("float i;" in self.insns[0].to_string(), "Printed insn has wrong contents")
        self.assertTrue("i=258f;" in self.insns[1].to_string(), "Printed insn has wrong contents")
