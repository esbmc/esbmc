# `id` has no static type and the formatting runs, so the path is refused
# where it runs rather than given a made-up string.
class Packet:
    def __init__(self, ident):
        self.ident = ident


class Task:
    def label(self, id):
        return "id %d" % id

    def queue(self, pkt):
        return self.label(pkt.ident)


assert Task().queue(Packet(3)) == ""
