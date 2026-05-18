import threading


class SharedResource:
    def __init__(self) -> None:
        self.mutex = threading.Lock()
        self.value: int = 0


def worker(resource: SharedResource) -> None:
    resource.mutex.acquire()
    resource.value = resource.value + 1
    resource.mutex.release()


# Canonical SharedResource pattern from the issue body of #4568 / #4583:
# two threads share a class instance carrying both a Lock and an int.
# Before #4583's fix, the args-struct field for the instance was
# declared as int (degraded from the construction-site rebind) and the
# trampoline's call to worker(arg) dereferenced an int as a pointer,
# producing "Access to object out of bounds" inside Lock's __init__.
resource = SharedResource()
t1 = threading.Thread(target=worker, args=(resource,))
t2 = threading.Thread(target=worker, args=(resource,))
t1.start()
t2.start()
t1.join()
t2.join()

assert resource.value == 2
