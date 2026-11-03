class Edge:
    def __init__(self, a, b):
        self.node = a
        self.cost = b


size = 5
adj = [[] for x in range(size)]
adj[0].append(Edge(1, 23))
adj[0].append(Edge(3, 7))
adj[3].append(Edge(1, -50))
assert isinstance(adj[0][0], Edge)
assert adj[0][1].cost == 7
assert adj[3][0].node == 1
assert len(adj[1]) == 0
