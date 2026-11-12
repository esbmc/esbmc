import random

use_int = random.random() < 0.5
if use_int:
    r = 12
else:
    r = "ab"
if not use_int:
    assert len(r) == 2
