# Falsification harness for threading.Lock mutual exclusion
# (src/python-frontend/models/threading.py).
#
# Under the Lock both increments land, so the final counter is 2, never 1.
# Asserting the smaller value must therefore be falsifiable — this is the
# non-vacuous counterpart to harness_threading_lock_mutex (it confirms the
# interleaving search actually drives counter to 2 rather than passing
# vacuously). Same flags: --context-bound 2 --data-races-check.
#
# WRONG PROPERTY (expected to be falsified):
#   F1: counter == 1 after two locked increments.  Mutual exclusion gives 2.
import threading

counter: int = 0
lock = threading.Lock()


def bump() -> None:
    global counter
    lock.acquire()
    tmp: int = counter
    counter = tmp + 1
    lock.release()


t1 = threading.Thread(target=bump)
t2 = threading.Thread(target=bump)
t1.start()
t2.start()
t1.join()
t2.join()

assert counter == 1  # F1 — falsifiable (locked increments give 2)
