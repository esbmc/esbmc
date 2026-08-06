# deque / defaultdict / OrderedDict are classes in Python but the operational
# model implements them as functions, so an annotation naming one used to be
# rejected with "NameError: name 'deque' is not defined". The annotation now
# resolves to the function's declared return type.
import collections
from collections import deque, OrderedDict

a: collections.deque = collections.deque()
a.append(3)
assert len(a) == 1

b: deque = deque()
b.append(4)
b.append(5)
assert len(b) == 2

c: collections.defaultdict = collections.defaultdict(int)
c["k"] = 1
assert c["k"] == 1

d: OrderedDict = OrderedDict()
d["k"] = 2
assert d["k"] == 2
