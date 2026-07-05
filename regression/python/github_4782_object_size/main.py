# Regression for #4782 / #4804 / #4805: a non-array pointer reaching the
# __ESBMC_get_object_size intrinsic (here via a set() of user objects in a
# graph traversal) used to dereference an empty/non-array deref item and crash
# — SIGSEGV in release, assertion failure in debug. #5658 replaced the crash
# (and the clean-diagnostic abort that followed it) with a nondet size model,
# so this case no longer aborts, but the recursive set-based DFS still hits a
# separate symbolic-list-size symex wall and times out — stays KNOWNBUG.
class Node:
    def __init__(self, value=None, successors=[]):
        self.value = value
        self.successors = successors


def depth_first_search(startnode, goalnode):
    nodesvisited = set()

    def search_from(node):
        if node in nodesvisited:
            return False
        elif node is goalnode:
            return True
        else:
            nodesvisited.add(node)
            return any(search_from(nextnode) for nextnode in node.successors)

    return search_from(startnode)


a = Node("A")
b = Node("B")
a.successors = [b]
assert depth_first_search(a, b)
