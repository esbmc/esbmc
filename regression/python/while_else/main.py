# Python while-else: the else clause runs when the loop condition becomes false
# (the loop ends without hitting break), and is skipped when the loop exits via
# break. This previously aborted conversion with "while takes two operands"
# because the else block was emitted as an invalid third operand of the GOTO
# while. It is now lowered with a did-not-break flag, the same desugaring used
# for for-else.

# No break: the else clause runs.
i = 0
while i < 3:
    i += 1
else:
    i = 100
assert i == 100

# break taken: the else clause is skipped.
found = -1
j = 0
while j < 5:
    if j == 2:
        found = j
        break
    j += 1
else:
    found = 100
assert found == 2
