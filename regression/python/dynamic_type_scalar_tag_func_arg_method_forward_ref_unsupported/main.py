class C:
    def a(self):
        cond = nondet_bool()
        if cond:
            x = 1
        else:
            x = "a"
        return self.b(x)

    def b(self, v):
        return v == 1

c = C()
c.a()
