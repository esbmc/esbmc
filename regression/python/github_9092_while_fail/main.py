# Issue #9092: a raising while condition followed by another property aborted.
class C:
    pass


c = C()
while c.limit:
    pass
x = 1 // 0
