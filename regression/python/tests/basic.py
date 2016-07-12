import unittest

class Basic(unittest.TestCase):
    def test_import(self):
        try:
            import esbmc
            self.assertTrue(True)
        except:
            self.assertTrue(False, "Failed to import esbmc")
