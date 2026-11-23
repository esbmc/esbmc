class Node:

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return 10 // (self.value - 1)
