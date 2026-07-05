# Negative variant of github_3067_return_reference: the returned object is a
# reference, so mutating it through one alias is observed through the other.
# The assertion below contradicts that and must be detected as a violation.
class Node:
    def __init__(self, value: int) -> None:
        self.value = value


def make(v: int) -> Node:
    return Node(v)


a = make(7)
b = a
b.value = 100
assert a.value == 7  # wrong: a aliases b, so a.value is 100
