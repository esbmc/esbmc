class Base:

    def __eq__(self, other):
        return 1 // 0 == 0


class Node(Base):

    def __init__(self, value):
        self.value = value
