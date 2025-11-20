
def possible_change(coins, total):
    if total == 0:
        return 1
    if total < 0 or not coins:
        return 0

    first, *rest = coins
    return possible_change(coins, total - first) + possible_change(rest, total)

"""
def possible_change(coins, total):
    if total == 0:
        return 1
    if not coins or total < 0:
        return 0

    first, *rest = coins
    return possible_change(coins, total - first) + possible_change(rest, total)

def possible_change(coins, total):
    if total == 0:
        return 1
    if total < 0 or len(coins) == 0:
        return 0

    first, *rest = coins
    return possible_change(coins, total - first) + possible_change(rest, total)

def possible_change(coins, total):
    if total == 0:
        return 1
    if len(coins) == 0 or total < 0:
        return 0

    first, *rest = coins
    return possible_change(coins, total - first) + possible_change(rest, total)

def possible_change(coins, total):
    if total == 0:
        return 1
    if not coins: return 0
    if total < 0:
        return 0

    first, *rest = coins
    return possible_change(coins, total - first) + possible_change(rest, total)

def possible_change(coins, total):
    if total == 0:
        return 1
    if len(coins) == 0: return 0
    if total < 0:
        return 0

    first, *rest = coins
    return possible_change(coins, total - first) + possible_change(rest, total)

"""

assert possible_change([1, 4, 2], -7) == 0
assert possible_change([1, 5, 10, 25], 11) == 4
assert possible_change([1, 5, 10, 25], 75) == 121
assert possible_change([1, 5, 10, 25], 34) == 18
assert possible_change([1, 5, 10], 34) == 16
assert possible_change([1, 5, 10, 25], 140) == 568
assert possible_change([1, 5, 10, 25, 50], 140) == 786
assert possible_change([1, 5, 10, 25, 50, 100], 140) == 817
assert possible_change([1, 3, 7, 42, 78], 140) == 981
assert possible_change([3, 7, 42, 78], 140) == 20

