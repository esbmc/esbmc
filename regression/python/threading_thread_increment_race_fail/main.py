import threading

# Two concurrent threads perform a read-modify-write on the same
# module-level global with no synchronisation. The classic interleaving
# where both threads read counter==0 before either writes leaves
# counter==1, violating the assertion.
#
# Regression for #4584. ESBMC reports the assertion violation under
# ``--data-races-check``: the flag both adds race-eligibility yields
# at every shared-global access and (via execution_statet::
# check_if_ileaves_blocked) keeps interleaving generation alive after
# __ESBMC_main has ended in some sibling schedule, which is what lets
# the DFS construct the race-exposing schedule, leaving the renamed
# value of counter at 1 (not 2) in the user-level claim.
#
# Without ``--data-races-check`` the assertion is still missed: the
# tree-global ``main_thread_ended`` flag fires while later schedules
# still have live spawned threads, pruning the race-exposing branches.
# A robust scheduler-side fix needs deeper changes to how that flag
# interacts with DFS backtracking (#4584 follow-up).
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
