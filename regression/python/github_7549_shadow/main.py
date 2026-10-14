# A name rebound to a value resolves to that value, even where a class of the
# same name exists: the class-object fallback must sit behind symbol lookup,
# not ahead of it (#7549).
class C:
    pass


class Item:
    pass


C = 5
assert C == 5

total = 0
for Item in range(3):
    total = total + Item

assert total == 3
