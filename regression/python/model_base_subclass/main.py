# Subclassing a class defined in an operational model used to abort: a
# qualified base (`queue.Queue`) parses as an Attribute, and the MRO walk read
# its "id" key, which only a bare Name carries.
import queue


class MyQ(queue.Queue):
    def put_two(self, a: int, b: int) -> None:
        self.put(a)
        self.put(b)


assert True
