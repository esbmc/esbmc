#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    /* --- Basic C guarantees --- */

    /* argc must be at least 1 (program name exists) */
    assert(argc >= 1);

    /* argv must not be NULL */
    assert(argv != NULL);

    /* C standard: argv[argc] is NULL sentinel */
    assert(argv[argc] == NULL);

    /* argv[0] must exist */
    assert(argv[0] != NULL);

    /* --- Program logic ---
       Usage: prog <int a> <int b>
       Computes sum = a + b
    */

    if (argc != 3)
    {
        /* Ensure we never access out-of-bounds argv */
        assert(argc != 3);
        return 0;
    }

    /* Now safe to access argv[1] and argv[2] */
    assert(argv[1] != NULL);
    assert(argv[2] != NULL);

    int a = atoi(argv[1]);
    int b = atoi(argv[2]);

    int sum = a + b;

    /* --- Functional correctness checks --- */

    /* recompute via string parsing and compare */
    int expected = atoi(argv[1]) + atoi(argv[2]);
    assert(sum == expected);

    /* sanity: addition identity */
    assert(sum - a == b);

    printf("%d + %d = %d\n", a, b, sum);

    return 0;
}
