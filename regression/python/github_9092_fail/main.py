# Issue #9092: an exception raised in an if condition was lost.
class C:
    pass


c = C()
if len(c) > 0:
    pass
