# The empty list is never popped: the walrus operand is guarded (#10105).
words: list[str] = []
while words and (w := words.pop()):
    pass
assert len(words) == 0
