import time

def run() -> None:
    start = time.time()
    time.sleep(1)
    end = time.time()

    assert start == end

run()
