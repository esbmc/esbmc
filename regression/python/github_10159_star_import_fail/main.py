# The star import rebinds len after the module-level def (#10159).
def len(v) -> int:
    return 3


from m import *

assert len([1]) == 3
