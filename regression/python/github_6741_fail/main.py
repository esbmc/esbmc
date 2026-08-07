def byteslike(*pos):
    return len(pos)


assert byteslike(1, 2) == 3
