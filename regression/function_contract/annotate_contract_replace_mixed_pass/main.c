/* Test: replace-all-contracts with mixed annotated/non-annotated functions
 *
 * add() is NOT annotated -> its body executes normally, returns a + b.
 * increment() IS annotated with contract -> its call is replaced.
 *
 * Only increment() should be replaced. add() should still run its body.
 */
#include <assert.h>

int counter = 0;

/* Not annotated: body executes normally */
int add(int a, int b)
{
    return a + b;
}

/* Annotated: call will be replaced with contract */
__attribute__((annotate("__ESBMC_contract")))
int increment(int x)
{
    __ESBMC_requires(x >= 0);
    __ESBMC_ensures(__ESBMC_return_value == x + 1);

    return x + 1;
}

int main(void)
{
    /* add() is not replaced, executes normally */
    int sum = add(3, 4);
    assert(sum == 7);

    /* increment() is replaced with contract: assume(ret == x + 1) */
    int result = increment(5);
    assert(result == 6);

    return 0;
}
