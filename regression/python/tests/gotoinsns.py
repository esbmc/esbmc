import unittest

class Gotoinsns(unittest.TestCase):
    def setUp(self):
        import esbmc
        # cwd = regression/python
        self.ns, self.opts, self.po = esbmc.init_esbmc_process(['test_data/00_big_endian_01/main.c', '--big-endian', '--bv'])
        self.funcs = self.po.goto_functions
        self.main = self.funcs.function_map[esbmc.irep_idt('main')].body
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
        self.assertTrue(code.value.as_string() == "main::main::1::i", "decl insn has wrong expr value")
        self.assertTrue(theinsn.function.as_string() == "main", "decl insn has wrong function name")

    def test_insn_locations(self):
        import esbmc
        theinsn = self.insns[0]
        loc = esbmc.location.from_locationt(theinsn.location)
        self.assertTrue(loc.file.as_string() == 'main.c', "File string is wrong")
        self.assertTrue(loc.function.as_string() == "main", "Func name in location is wrong")
        self.assertTrue(loc.line == 5, "Line number in test file is wrong")
        self.assertTrue(loc.column == 0, "Column number in test file is wrong")

    def test_more_fields(self):
        import esbmc
        theinsn = self.insns[0]
        self.assertTrue(esbmc.downcast_expr(theinsn.guard).value == True, "insn guard should be true")
        # No access to labels field right no
        # This has the value 99 right now, but we can't really assert that
        # because it'll change when we edit... anything
        self.assertTrue(theinsn.location_number > 0, "Wrong insn number")
        self.assertTrue(theinsn.loop_number == 0, "Loop number doesn't exist?")
        # This seems to be a useless field
        self.assertTrue(theinsn.target_number == 4294967295, "Target number doesn't exist?")

        norm = theinsn.to_string()
        wloc = theinsn.to_string(True)
        self.assertTrue(norm != wloc, "var changing flags should change to_string output")

        self.assertTrue(theinsn.is_other(), "is_other method of insn should work")

    def test_insn_clear(self):
        import esbmc
        self.insns[0].clear(esbmc.goto_programs.goto_program_instruction_type.ASSIGN)
        self.assertTrue(self.insns[0].type == esbmc.goto_programs.goto_program_instruction_type.ASSIGN, "Cleared insn should be assign")
        self.assertTrue(self.insns[0].code == None, "Cleared insn should have nil irep")

    def test_insn_printing(self):
        self.assertTrue("float i;" in self.insns[0].to_string(), "Printed insn has wrong contents")
        self.assertTrue("i=258f;" in self.insns[1].to_string(), "Printed insn has wrong contents")

    def test_insn_targets(self):
        import esbmc
        # First branch insn...
        self.assertTrue(self.insns[5].target != None, "6th insn in main should be branch")
        self.assertTrue(type(self.insns[5].target) == esbmc.goto_programs.instructiont, "Branch target should be insn")
        self.assertTrue(self.insns[5].target in self.insns, "Branch target should be in same program!")

    def test_insn_target_munging(self):
        # Check that if we change the target of a brnach that it's reflected
        # after a to/from goto program conversion
        import esbmc
        self.assertTrue(self.insns[0].target == None, "First main insn should not be a branch")
        # Self loop, slightly illegitmate as this isn't a branch insn
        self.insns[0].target = self.insns[0]
        self.main.set_instructions(self.insns)
        insns2 = self.main.get_instructions()
        self.assertTrue(insns2[0].target is insns2[0], "Branch target not mirrored after get/set insns")
