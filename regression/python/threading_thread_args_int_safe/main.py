import threading


result: int = 0


def setter(x: int) -> None:
    global result
    result = x


t = threading.Thread(target=setter, args=(42,))
t.start()
t.join()

assert result == 42
