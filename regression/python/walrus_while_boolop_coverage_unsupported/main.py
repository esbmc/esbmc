# Coverage modes convert every and/or operand unguarded, so a walrus there
# would pop the empty list; ESBMC refuses it.
words: list[str] = []
while words and (w := words.pop()):
    pass
assert len(words) == 0
