# isinstance() on an Optional[int] value answers from its is_none flag (#9044)
import random

n = random.randint(0, 10)
v = n if n > 0 else None
assert isinstance(v, int) or v is None
assert isinstance(v, int) == (n > 0)
assert isinstance(v, type(None)) == (n == 0)
assert not isinstance(v, str)
