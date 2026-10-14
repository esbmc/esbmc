# list.count(x) and list.index(x). Previously these were misrouted to the
# string handler (__python_str_count) and produced a non-deterministic result,
# so even a correct assertion failed. They now walk the list comparing elements.
a = [1, 2, 2, 3]
assert a.count(2) == 2
assert a.count(9) == 0
assert a.count(1) == 1

b = [10, 20, 30]
assert b.index(20) == 1
assert b.index(10) == 0

# first match wins; string elements; use in a larger expression
c = [5, 7, 5]
assert c.index(5) == 0
d = ["a", "b", "a"]
assert d.count("a") == 2
assert d.count("a") + d.index("b") == 3


def via_param(lst: list) -> int:
    return lst.count(2)


assert via_param([2, 2, 2]) == 3

# Dispatch guard: str and tuple count/index must keep routing to their own
# handlers (the list change defers only list/tuple receivers out of the string
# handler, never str receivers).
s = "abcabc"
assert s.count("a") == 2

t = (1, 2, 2, 3)
assert t.count(2) == 2
assert t.index(2) == 1
