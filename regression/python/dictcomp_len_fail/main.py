# Soundness guard for the dict-comprehension len() fix (#5222): an assertion
# that states the wrong size must still FAIL. This rules out a vacuous fix
# where len() of a dict-comprehension result is treated as a constant that
# satisfies any equality.
d = {i: i for i in range(2)}
assert len(d) == 5
