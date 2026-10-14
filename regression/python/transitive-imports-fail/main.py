from stubs import make_tile
from kernels.kernel import add_kernel

# Mismatched shapes — check_shape (imported transitively through kernel)
# must fire its assertion, proving the transitive import chain is wired up.
a = make_tile(4, 8)
b = make_tile(8, 4)
add_kernel(a, b)
