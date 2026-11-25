from heapq import heappush


def get(node_heap, wanted):
    for dist, node in node_heap:
        if node == wanted:
            return dist
    return 0


def run():
    h = []
    heappush(h, (7, 1))
    heappush(h, (2, 3))
    assert get(h, 3) == 7


run()
