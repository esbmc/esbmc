import threading

# End-to-end exercise of threading.Lock + threading.Thread lowering: two
# threads acquire (mutex, lock) in the SAME order before incrementing the
# shared `counter`, and the final value is exactly 2.
#
# NOTE: ESBMC currently marks module-level Python globals with
# static_lifetime=false, so rw_set/execution_state under-approximate
# races on `counter`. The SUCCESSFUL verdict here validates the
# threading.Thread + Lock lowering and the join-happens-before semantics
# — it does NOT prove the locks enforce mutual exclusion in the model.
# Follow-up issues track the static_lifetime gap and the parallel
# Lock-vs--deadlock-check gap.
mutex = threading.Lock()
lock = threading.Lock()
counter: int = 0


def thread_a() -> None:
    global counter
    mutex.acquire()
    lock.acquire()
    counter = counter + 1
    lock.release()
    mutex.release()


def thread_b() -> None:
    global counter
    mutex.acquire()
    lock.acquire()
    counter = counter + 1
    lock.release()
    mutex.release()


t1 = threading.Thread(target=thread_a)
t2 = threading.Thread(target=thread_b)
t1.start()
t2.start()
t1.join()
t2.join()

assert counter == 2
