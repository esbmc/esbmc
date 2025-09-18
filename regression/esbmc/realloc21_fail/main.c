#include <assert.h>
#include <stdlib.h>

/*
$ clang -fsanitize=address -g -o test_bug main.c
steps = 9;
unsigned shrink_step = 0;
test_bug: main.c:32: void test_alternating_resize(): Assertion `vec != NULL' failed.
Aborted (core dumped)
*/

//unsigned nondet_uint();

void test_alternating_resize()
{
    unsigned steps = nondet_uint();
    __VERIFIER_assume(steps > 0 && steps <= 10); // symbolic upper bound

    unsigned n = 0;
    unsigned capacity = 1;
    int *vec = malloc(capacity * sizeof(int));
    assert(vec != NULL);

    for (unsigned i = 0; i < steps; ++i)
    {
        // Grow step
        if (n == capacity)
        {
            capacity *= 2;
            vec = realloc(vec, capacity * sizeof(int));
            assert(vec != NULL);
        }
        vec[n++] = i;
        unsigned shrink_step = nondet_uint() % 2;
        // Shrink step
        if (n > 0 && (shrink_step == 0)) // symbolic choice to shrink
        {
            n--;
            vec = realloc(vec, n * sizeof(int));
            assert(vec != NULL);
        }
    }

    free(vec);
}

int main()
{
    test_alternating_resize();
    return 0;
}

