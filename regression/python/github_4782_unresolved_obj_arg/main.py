# Regression for #4782: passing an object whose type could not be resolved
# (an element of an untyped list, lowered to a void pointer) to a function
# whose formal parameter is a struct used to abort ESBMC with a core dump in
# goto_symext::argument_assignments. It must now surface as a clean diagnostic.
class Node:
    def __init__(self, nxts=[]):
        self.nxts = nxts


def visit(n):
    return any(visit(x) for x in n.nxts)


def main():
    a = Node()
    b = Node([a])
    assert visit(b) == False


main()
