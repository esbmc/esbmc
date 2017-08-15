import unittest
import functools

class Gotoprogs(unittest.TestCase):
    def setUp(self):
        import esbmc
        # cwd = regression/python
        self.ns, self.opts, self.po = esbmc.init_esbmc_process(['test_data/00_big_endian_01/main.c', '--big-endian', '--bv'])
        self.funcs = self.po.goto_functions
        self.main = self.funcs.function_map[esbmc.irep_idt('main')].body

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

    def test_empty(self):
        self.assertTrue(not self.main.empty(), "Main function shouldn't be empty")

    def test_empty2(self):
        import esbmc
        other_func = self.funcs.function_map[esbmc.irep_idt('__ESBMC_atomic_begin')].body
        self.assertTrue(other_func.empty(), "Atomic begin should be empty")

    def test_update(self):
        insns = self.main.get_instructions()
        self.assertTrue(insns[1].location_number != 0, "main 2nd insn should not be number zero")
        insns[1].location_number = 0
        self.main.set_instructions(insns)
        self.main.update()

        insns = self.main.get_instructions()
        self.assertTrue(insns[1].location_number != 0, "main 2nd insn should not be number zero after reset")

    def test_insns_clear(self):
        import esbmc
        self.main.clear()
        self.assertTrue(len(self.main.get_instructions()) == 0, "Cleared program should have no insns")

    def test_throw(self):
        import esbmc
        insns = self.main.get_instructions()
        # At least one of these should have a target
        insn_w_target = functools.reduce(lambda x, y: y if y.target != None else x, insns, None)
        self.assertTrue(insn_w_target != None, "Should have insn with target somewhere")
        insn_w_target.target = esbmc # Obviously non-insn target
        try:
            self.main.set_instructions(insns)
        except RuntimeError:
            pass
        except TypeError: # New in python3
            pass
        else:
            self.assertTrue(False, "set_instructions should throw on bad target")
