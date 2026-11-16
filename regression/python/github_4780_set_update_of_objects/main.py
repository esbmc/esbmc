class Node:
    def __init__(self, successors=[]):
        self.successors = successors


s1 = Node()
s2 = Node([s1])
queue = [s2]
seen = set()
while queue:
    node = queue.pop(0)
    queue.extend(node.successors)
    seen.update(node.successors)
assert len(seen) == 1
