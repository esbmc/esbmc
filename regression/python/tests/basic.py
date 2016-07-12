import unittest

class Basic(unittest.TestCase):
    def test_import(self):
        try:
            import esbmc
            self.assertTrue(True)
        except:
            self.assertTrue(False, "Failed to import esbmc")

    def test_types(self):
        import esbmc
        # Ensure that the base list of types are in the type class
        base_types = ['array', 'bool', 'code', 'cpp_name', 'empty', 'fixedbv',\
                'pointer', 'signedbv', 'string', 'struct', 'symbol', 'type2t',\
                'type_ids', 'type_vec', 'union', 'unsignedbv']
        for x in base_types:
            self.assertTrue(x in dir(esbmc.type), "Missing type {}".format(x))
