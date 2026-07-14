# Negative variant of while_else: the loop exits via break, so the else clause
# must be skipped. Asserting that the else ran must FAIL -- confirms the
# did-not-break flag correctly suppresses the else after a break (not vacuously
# accepted).

x = 0
while True:
    x = 5
    break
else:
    x = 99

assert x == 99
