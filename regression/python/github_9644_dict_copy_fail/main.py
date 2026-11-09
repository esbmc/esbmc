# GitHub #9644: iterating the copy yields its keys, so k takes the value 2.
mt = {1: "hi", 2: "yo"}
mt2 = mt.copy()
for k in mt2:
    assert k != 2
