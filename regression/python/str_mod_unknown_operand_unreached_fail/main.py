# `id` has no static type. The formatting runs only for an id above 5, which
# never happens here, so the refusal must not stop the program.
class Packet:
    def __init__(self, ident):
        self.ident = ident


class Task:
    def find(self, id):
        if id > 5:
            raise Exception("Bad task id %d" % id)
        return id

    def queue(self, pkt):
        return self.find(pkt.ident)


assert Task().queue(Packet(3)) == 4
