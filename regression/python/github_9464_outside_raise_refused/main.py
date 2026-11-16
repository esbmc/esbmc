# Outside a raise the text of type() would be observable, so it stays refused.
class V:
    def name(self, other):
        return str(type(other))


v = V()
c = nondet_bool()
other = V() if c else 1
s = v.name(other)
