# github.com/esbmc/esbmc/issues/6242
# The factory method lives in a BASE class, so typing self.attr from
# self.make() must walk the base chain (not just the enclosing class) to
# adopt the return class; otherwise the later attribute method call aborts.
class Publisher:
    def publish(self, msg):
        return msg + 1


class Base:
    def make(self):
        return Publisher()


class Talker(Base):
    def __init__(self):
        self.pub = self.make()

    def go(self):
        return self.pub.publish(41)


assert Talker().go() == 42
