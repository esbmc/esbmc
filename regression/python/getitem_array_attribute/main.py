from array import array


class Grid:
    def __init__(self):
        self.data = array("d", [0.0, 0.0, 0.0, 0.0])


assert Grid().data[3] == 0.0
