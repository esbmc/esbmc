# The formatting of an operand with no static type sits in a ternary branch
# and an `or` tail that do not run, so neither refusal is reached.
class Packet:
    def __init__(self, ident):
        self.ident = ident


class Task:
    def label(self, id, ok):
        return "fine" if ok else "id %d" % id

    def tail(self, id, ok):
        return ok or "id %d" % id == ""

    def queue(self, pkt):
        return self.label(pkt.ident, True)

    def check(self, pkt):
        return self.tail(pkt.ident, True)


assert Task().queue(Packet(3)) == "fine"
assert Task().check(Packet(3))
