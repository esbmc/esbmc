def get(node_heap, wanted):
    for dist, node in node_heap:
        if node == wanted:
            return dist
    return 0


h: list[tuple[int, int]] = []
h.append((7, 1))
assert get(h, 1) == 7
