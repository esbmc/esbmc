from collections import deque as Queue
from node import Node


def breadth_first_search(startnode, goalnode):
    queue = Queue()
    queue.append(startnode)

    nodesseen = set()
    nodesseen.add(startnode)

    while queue:
        node = queue.popleft()

        if node is goalnode:
            return True
        else:
            queue.extend(node for node in node.successors if node not in nodesseen)
            nodesseen.update(node.successors)

    return False


station1 = Node(1)
station2 = Node(2, None, [station1])

assert breadth_first_search(station2, station1)
