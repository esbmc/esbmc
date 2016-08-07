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

    def test_atomic_access(self):
        import esbmc
        self.assertTrue(len(self.curstate.atomic_numbers) > 0, "Should have at least one atomic number")
        foo = self.curstate.atomic_numbers[0]
        self.assertTrue(type(foo) == int, "Atomic numbers should contain ints")

    def test_l2_access(self):
        import esbmc
        foo = self.curstate.state_level2
        self.assertTrue(type(foo) == esbmc.symex.execution_state.ex_state_level2t, "Can't acces level2 state")

    def test_value_set_access(self):
        import esbmc
        foo = self.curstate.global_value_set
        self.assertTrue(type(foo) == esbmc.value_set, "Can't acces global value_set")

    # XXX testing: there are no other complex types that we want to actually
    # test in execution_statet (vector of sets is too tricky; vector of vectors
    # is part of mpor which I don't want to expose).
    # I also don't really want to test the functionality of ex_state too much,
    # we know we can call through to it, and what we're testing here is the
    # python interface not the actual mechanics of esbmc.
    # Thus, the final thing to test is our ability to override virtual functions
    # and install our own python functions on top.
