/*
 * C equivalent of func_internal_type_1 (Solidity Pyramid example).
 * Tests: function pointers as parameters, indirect calls in loops.
 *
 * Solidity: function (uint) pure returns (uint) f  =>  C: unsigned (*f)(unsigned)
 *
 * If ESBMC backend handles function pointers correctly, this should
 * produce VERIFICATION SUCCESSFUL (both assertions hold).
 */
#include <assert.h>

#define MAX_LEN 8

/* --- ArrayUtils equivalent (stack-allocated) --- */

void map(unsigned *self, unsigned len,
         unsigned (*f)(unsigned),
         unsigned *result)
{
    for (unsigned i = 0; i < len; i++) {
        result[i] = f(self[i]);
    }
}

unsigned reduce(unsigned *self, unsigned len,
                unsigned (*f)(unsigned, unsigned))
{
    unsigned r = self[0];
    for (unsigned i = 1; i < len; i++) {
        r = f(r, self[i]);
    }
    return r;
}

void range(unsigned length, unsigned *result)
{
    for (unsigned i = 0; i < length; i++) {
        result[i] = i;
    }
}

/* --- Pyramid contract equivalent --- */

unsigned square(unsigned x) { return x * x; }
unsigned sum(unsigned x, unsigned y) { return x + y; }

unsigned pyramid(unsigned l)
{
    unsigned r[MAX_LEN];
    range(l, r);

    unsigned mapped[MAX_LEN];
    map(r, l, square, mapped);

    return reduce(mapped, l, sum);
}

int main()
{
    /* pyramid(4) = [0,1,2,3] -> [0,1,4,9] -> 14 */
    assert(pyramid(4) == 14);

    /* pyramid(1) = [0] -> [0] -> 0 */
    assert(pyramid(1) == 0);

    return 0;
}
