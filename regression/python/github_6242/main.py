# github.com/esbmc/esbmc/issues/6242
# An attribute assigned from a self-method returning an object must be typed by
# the method's return class, so a later method call on the attribute resolves
# instead of aborting GOTO conversion.
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


assert Talker().go() == 42
