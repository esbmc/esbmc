import threading

flag: int = 0


def setter() -> None:
    global flag
    flag = 1


t = threading.Thread(target=setter)
t.start()
# Read BEFORE join: an interleaving in which `setter` has not yet
# executed must be feasible, so the assertion is violated. This is the
# negative counterpart to threading_thread_join_orders_writes_safe and
# confirms that ESBMC actually explores the spawn-vs-main interleaving.
assert flag == 1
t.join()
