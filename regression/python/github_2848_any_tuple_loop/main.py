from typing import Any

# Rebinding inside a loop body refuses the fresh-alias slot, so the
# in-place-retype backstops must keep the symbol tuple-typed; reads
# emitted before the loop stay valid (no member-of-scalar at migration).
x: Any = (1, 2)
y = x[0]
i = 0
while i < 3:
    x = 5
    i = i + 1
assert y == 1
