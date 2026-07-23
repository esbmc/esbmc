# github.com/esbmc/esbmc/issues/6242 (review finding #4)
# The issue's original reproducer: an attribute assigned from a method whose
# return type is NOT the enclosing class, then a method call on that attribute.
# This used to core-dump (SIGABRT); it must now degrade to a clean verdict.
class Talker:
    def make(self) -> int:
        return 5

    def __init__(self):
        self.pub = self.make()

    def go(self):
        self.pub.publish(1)


Talker().go()
