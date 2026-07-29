# The drain added for #6489 must not mask a genuine deadlock: main releases m0
# first (running the drain), and only then do the workers deadlock on m1/m2.
import threading

m0 = threading.Lock()
m1 = threading.Lock()
m2 = threading.Lock()


def t1() -> None:
    m1.acquire()
    m2.acquire()
    m2.release()
    m1.release()


def t2() -> None:
    m2.acquire()
    m1.acquire()
    m1.release()
    m2.release()


m0.acquire()
m0.release()
a = threading.Thread(target=t1)
b = threading.Thread(target=t2)
a.start()
b.start()
a.join()
b.join()
