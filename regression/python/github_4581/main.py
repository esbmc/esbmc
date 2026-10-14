import threading

lock_a = threading.Lock()
lock_b = threading.Lock()


def thread_a() -> None:
    lock_a.acquire()
    lock_b.acquire()
    lock_b.release()
    lock_a.release()


def thread_b() -> None:
    lock_b.acquire()
    lock_a.acquire()
    lock_a.release()
    lock_b.release()


t1 = threading.Thread(target=thread_a)
t2 = threading.Thread(target=thread_b)
t1.start()
t2.start()
t1.join()
t2.join()
