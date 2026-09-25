# A tuple variable keeps its first binding's type when rebound, so its
# truthiness is refused rather than read from that type.
x = (1, )
x = ()
assert not bool(x)
