# isinstance() on an Optional[int] value answers from its is_none flag (#9044)
import random

n = random.randint(1, 10)
v = n if n > 0 else None
assert not isinstance(v, int)
