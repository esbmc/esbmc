counter = 0


def bump() -> int:
    global counter
    counter += 1
    return counter


def size(board):
    return board.size


class Board:
    first = bump()

    def __init__(self):
        self.size = 2


assert size(Board()) == 2
assert Board.first == 1
assert counter == 1
