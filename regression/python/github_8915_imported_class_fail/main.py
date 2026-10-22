# #8915: as github_8915_unresolved_method_side_effect_fail, with the class defined
# in an imported module.
from sq import Square


def useful(square):
    for neighbour in square.neighbours:
        r = neighbour.head()


a = Square()
a.neighbours.append(a)
useful(a)
assert a.timestamp == 1
