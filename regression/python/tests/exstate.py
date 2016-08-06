import unittest

class Exstate(unittest.TestCase):
    def setUp(self):
        import esbmc
        # cwd = regression/python
        self.ns, self.opts, self.po = esbmc.init_esbmc_process(['test_data/00_big_endian_01/main.c', '--big-endian', '--bv'])
        self.funcs = self.po.goto_functions
        self.main = self.funcs.function_map[esbmc.irep_idt('c::main')].body
        self.eq = esbmc.symex.equation(self.ns)
        self.art = esbmc.symex.reachability_tree(self.po.goto_functions, self.ns, self.opts, self.eq, self.po.context, self.po.message_handler)
        # Step that art once...
        self.art.setup_for_new_explore()
        self.art.get_cur_state().symex_step(self.art)
        self.curstate = self.art.get_cur_state()

    def tearDown(self):
        import esbmc
        esbmc.kill_esbmc_process()
        self.ns, self.opts, self.funcs = None, None, None
        self.main = None
        self.eq = None
        self.art = None

    def test_setup(self):
        # Does any of the above throw?
        pass

    def test_state_access(self):
        import esbmc
        self.assertTrue(len(self.curstate.threads_state) > 0, "Should have at least one thread")
        foo = self.curstate.threads_state[0]
        self.assertTrue(type(foo) == esbmc.symex.goto_symex_statet, "Thread state has incorrect type?")

    def test_state_append(self):
        import esbmc
        self.assertTrue(len(self.curstate.threads_state) > 0, "Should have at least one thread")
        foo = self.curstate.threads_state[0]
        self.curstate.threads_state.append(foo) # Is just a boost shared ptr reference
        self.assertTrue(len(self.curstate.threads_state) == 2, "Should have two g_s_s's after appending")
