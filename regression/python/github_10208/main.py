# A list that is changed after its binding has no statically known size.
buf = [0, 0]
buf.append(1)
b = bytes(buf)
assert len(b) == 3
