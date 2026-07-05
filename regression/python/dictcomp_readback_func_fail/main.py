# Soundness guard for the comprehension read-back fix (#5222): a wrong
# read-back value must still FAIL, ruling out a vacuous fix.


def from_range():
    d = {i: 7 for i in range(3)}
    return d[0]


assert from_range() == 999
