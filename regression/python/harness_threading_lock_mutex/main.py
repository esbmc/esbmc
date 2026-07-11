# Verification harness for threading.Lock mutual exclusion
# (src/python-frontend/models/threading.py).
#
# Two threads each perform a read-modify-write on a shared global while holding
# the same Lock. Because acquire()/release() serialise the critical section,
# both increments land and the final counter is exactly 2 in every interleaving
# — the Lock provides mutual exclusion. (Without the Lock this is the classic
# lost-update race; see regression/python/threading_thread_increment_race_fail.)
#
# The proof requires exploring thread interleavings: run with --context-bound 2
# and --data-races-check (the flag that keeps interleaving generation alive;
# without it the search terminates early and the claim passes vacuously).
#
# ENSURES:
#   E1: under the Lock, counter == 2 in every schedule [mutual exclusion holds]
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

assert counter == 2       # E1
