import threading

# Two concurrent threads write to the same module-level global with no
# synchronisation. Under --data-races-check ESBMC must report a W/W
# data race on `shared`.
shared: int = 0


def writer_a() -> None:
    global shared
    shared = 1


def writer_b() -> None:
    global shared
    shared = 2


t1 = threading.Thread(target=writer_a)
t2 = threading.Thread(target=writer_b)
t1.start()
t2.start()
t1.join()
t2.join()
