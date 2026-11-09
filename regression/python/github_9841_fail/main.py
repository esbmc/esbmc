# GitHub #9841: a literal passed by self.g(...) types g's parameter.
class M:
    def g(self, tnow):
        return int(tnow - 1.5)

    def run(self):
        return self.g(2.5)


assert M().run() == 7
