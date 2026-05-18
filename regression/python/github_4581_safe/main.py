import threading

# Single-threaded Lock acquire/release under --deadlock-check.
# Exercises both the if-branch (lock free → take it) and the
# unlock-with-held assertion of the deadlock-aware model without
# introducing thread contention, so the verdict is SUCCESSFUL by
# construction of the model alone.
lock = threading.Lock()
counter: int = 0


def bump() -> None:
    global counter
    lock.acquire()
    counter = counter + 1
    lock.release()


bump()
bump()
bump()

assert counter == 3
