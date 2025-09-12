
#include <stdint.h>
#include <assert.h>
#include <stddef.h>
#include <limits.h>
int main(void) {
        struct S { int x; int y; int z; } s = { .z = 42 };
        
        // TODO: This test case demonstrates two issues:
        // 1. UNDEFINED BEHAVIOR: Pointer arithmetic &s.x + 8 goes beyond bounds of s.x object (C11 6.5.6/8)
        // 2. ESBMC LIMITATION: Symbolic execution struggles with uintptr_t arithmetic patterns
        // The code computes &s.x + 8 which happens to equal &s.z on most systems, 
        // but violates C's object model since it performs arithmetic beyond the s.x object bounds.
        
        uintptr_t v = (uintptr_t)&s.x;
        uintptr_t u = offsetof(struct S, y);
        u *= 2;
        int *p = (int *)(u + v);
        *p = 3;
        assert(&s.z == p);
        assert(s.z == 3);
}

