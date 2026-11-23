from collections import deque as Queue


class Node:
    def __init__(self, value=None, successors=[]):
        self.value = value
        self.successors = successors


def breadth_first_search(startnode, goalnode):
    queue = Queue()
    queue.append(startnode)

    nodesseen = set()
    nodesseen.add(startnode)

    while True:
        node = queue.popleft()

        if node is goalnode:
            return True
        else:
            queue.extend(node for node in node.successors if node not in nodesseen)
            nodesseen.update(node.successors)

    return False


nodef = Node(1)
nodee = Node(2)

assert not breadth_first_search(nodef, nodee)
