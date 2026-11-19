# #10143: g() is "", so g() and True is "", which a void * result reads as true.
def g() -> str:
    return ""


z = g() and True
assert not z
