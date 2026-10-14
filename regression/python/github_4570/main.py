import threading


class SharedResource:
    def __init__(self):
        self.mutex = threading.Lock()


def use_lock(resource):
    resource.mutex.acquire()
    resource.mutex.release()


resource = SharedResource()
use_lock(resource)
