# github.com/esbmc/esbmc/issues/6242
# Same attribute-from-self-method pattern, but the asserted result is wrong:
# the call now resolves and returns a concrete value, so the assertion is
# checked and violated (previously this aborted with a core dump).
class Publisher:
    def publish(self, msg):
        return msg + 1


class Talker:
    def make(self):
        return Publisher()

    def __init__(self):
        self.pub = self.make()

    def go(self):
        return self.pub.publish(41)


assert Talker().go() == 99
