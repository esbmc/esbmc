class Packet:

    def __init__(self, i):
        self.ident = i


def find(i):
    return "id %d" % i


def qpkt(pkt):
    return find(pkt.ident)


def local():
    p = Packet(2)
    return find(p.ident)


assert qpkt(Packet(1)) == "id 1"
assert local() == "id 1"
