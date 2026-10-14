# A list of objects built by repetition (`[Obj()] * n`) then filled by index
# must still type the for-loop element as the object's class, so a method call
# on it resolves. Previously the loop variable fell back to Any (void*) and the
# call crashed GOTO generation with "Function ... not found".
class Point:
    def __init__(self, i):
        self.x = i

    def value(self):
        return self.x * 2


n = 3
pts = [Point(0)] * n
for i in range(n):
    pts[i] = Point(i)

total = 0
for p in pts:
    total += p.value()

assert total == 6  # (0 + 1 + 2) * 2
