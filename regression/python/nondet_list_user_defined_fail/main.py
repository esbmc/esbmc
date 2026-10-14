# Companion to nondet_list_user_defined: the user's function returns an empty
# list, so a non-empty one is not reachable and this assertion is falsified.
def nondet_list(n: int) -> list:
    return []

x: list = nondet_list(3)
assert len(x) == 1
