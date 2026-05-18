import threading

# Two concurrent threads perform a read-modify-write on the same
# module-level global with no synchronisation. The classic interleaving
# where both threads read counter==0 before either writes leaves
# counter==1, violating the assertion. ESBMC explores that interleaving
# and reports VERIFICATION FAILED.
#
# Regression for #4584. Building on PR #4587 (which taught
# execution_statet::get_expr_globals to mark Python module globals as
# race-eligible), the scheduler-side change gates the
# ``main_thread_ended`` pruning in check_if_ileaves_blocked on the
# current schedule's own main thread having ended — not on the
# tree-global flag set the first time any schedule's __ESBMC_main ran
# to END_FUNCTION. The Python frontend's user code lives in
# python_user_main (called by __ESBMC_main between two cleanup hooks),
# so the global flag was firing while later-explored schedules still
# had live spawned threads, pruning the race-exposing branches and
# leaving the renamed value of counter at 2 in the user-level claim.
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
