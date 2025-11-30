// Test for GitHub issue #3191 - C11 atomic types support
// This test should pass - correct use of atomics
#include <stdatomic.h>
#include <assert.h>

_Atomic(int) global_counter = 0;

int main() {
    // Test NonAtomicToAtomic cast during initialization
    _Atomic(int) x = 42;

    // Test NonAtomicToAtomic cast during assignment
    _Atomic(int) y;
    y = 10;

    // Test AtomicToNonAtomic cast
    int z = y;

    assert(x == 42);
    assert(y == 10);
    assert(z == 10);

    return 0;
}
