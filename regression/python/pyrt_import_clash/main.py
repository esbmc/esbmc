# Both this module and util bind `helper`. One namespace cannot hold both, so
# the import is refused rather than one of them silently winning.
def helper(a):
    return a + 9


import util

assert util.helper(1) == 2
