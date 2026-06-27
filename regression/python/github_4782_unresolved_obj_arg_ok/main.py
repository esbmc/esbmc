# Companion to github_4782_unresolved_obj_arg: when the object's type IS
# resolved, a pointer argument bound to a struct parameter (Python pass-by-
# reference) must still be dereferenced and verified normally.
class Node:

    def __init__(self, v: int):
        self.v = v


def get_v(n: Node) -> int:
    return n.v


def main():
    a = Node(5)
    assert get_v(a) == 5


main()
