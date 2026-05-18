import threading

# Two concurrent threads perform a read-modify-write on the same
# module-level global with no synchronisation. The classic interleaving
# where both threads read counter==0 before either writes leaves
# counter==1, violating the assertion. ESBMC should explore that
# interleaving and report VERIFICATION FAILED.
#
# Tracked as #4584: execution_statet::get_expr_globals already recognises
# Python module globals as shared (PR #4587), but the scheduler
# under-explores the bad schedule (13 interleavings vs the equivalent C
# program's 2604). KNOWNBUG until the scheduler-side fix lands.
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
