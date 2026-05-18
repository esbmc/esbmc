import threading


shared: int = 0


def writer(x: int) -> None:
    global shared
    shared = x


t = threading.Thread(target=writer, args=(7,))
t.start()
t.join()

# Read AFTER join: the join establishes happens-before, so the spawned
# thread's write to `shared` must be visible here.
assert shared == 7
