# Python translation of regression/esbmc-unix/github_6474/main.c: deadlock-free.
import threading

m1 = threading.Lock()
m2 = threading.Lock()


def w() -> None:
    m2.acquire()
    m1.acquire()
    m1.release()
    m2.release()


a = threading.Thread(target=w)
b = threading.Thread(target=w)
m1.acquire()
a.start()
b.start()
m1.release()
m2.acquire()
m2.release()
a.join()
b.join()
