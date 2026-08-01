import threading

# Two concurrent threads perform a read-modify-write on the same
# module-level global with no synchronisation. The classic interleaving
# where both threads read counter==0 before either writes leaves
# counter==1, violating the assertion.
#
# Regression for #4584, the plain path: no --data-races-check. That flag
# used to be the only thing keeping interleaving generation alive once a
# sibling schedule had run __ESBMC_main to completion, because the
# reachability tree tracked that per search rather than per state.
counter: int = 0


def bump() -> None:
    global counter
    tmp: int = counter
    counter = tmp + 1


t1 = threading.Thread(target=bump)
t2 = threading.Thread(target=bump)
t1.start()
t2.start()
t1.join()
t2.join()
assert counter == 2
