import threading


class SharedResource:
    def __init__(self):
        self.mutex = threading.Lock()
        self.flag: int = 0


def set_flag(resource):
    resource.mutex.acquire()
    resource.flag = 1
    resource.mutex.release()


resource = SharedResource()
set_flag(resource)
assert resource.flag == 2
