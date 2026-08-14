# Instantiating a subclass of a class defined in an operational model, then
# calling an inherited method through it. The constructor path read the base's
# "id" key, which only a bare Name carries -- a qualified base is an Attribute.
import queue


class MyQ(queue.Queue):
    pass


q = MyQ()
q.put(7)
assert q.get() == 7
