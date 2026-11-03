class Packet:

    def __init__(self, i):
        self.ident = i


def find(i):
    return "id %d" % i


assert find(Packet(1).ident) == "id 1"
