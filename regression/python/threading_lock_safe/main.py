import threading

counter: int = 0
lock = threading.Lock()


def bump() -> None:
    global counter
    lock.acquire()
    counter = counter + 1
    lock.release()


bump()
bump()
bump()

assert counter == 3
