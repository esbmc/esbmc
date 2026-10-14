from threading import Lock

held = Lock()
held.acquire()
held.release()
held.acquire()
held.release()

assert True
