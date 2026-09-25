# Queue.join() takes no arguments, but every attribute call is offered to the
# string handler first, and its str.join arm used to throw on any arity other
# than one instead of declining. The receiver never reached instance dispatch.
# Each put() is matched by a task_done() so join() returns under CPython too.
import queue
from queue import Queue

a: Queue = Queue()
a.put(7)
a.task_done()
a.join()
assert a.qsize() == 1

# Same call through a dotted annotation.
b: queue.Queue = queue.Queue()
b.put(8)
b.task_done()
b.join()
assert b.qsize() == 1

# str.join is unaffected.
assert ",".join(["x", "y"]) == "x,y"
