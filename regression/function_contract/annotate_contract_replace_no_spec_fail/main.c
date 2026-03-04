/* Test: replace-all-contracts with no contract specification
 *
 * Function is annotated but has no requires/ensures/assigns.
 * In replace mode, the function body is removed and all globals
 * are conservatively havoced. The assert on x should FAIL because
 * x is havoced to a nondeterministic value.
 */
#include <assert.h>

int x = 0;

__attribute__((annotate("__ESBMC_contract")))
void do_something(void)
{
    x = x + 1;
}

int main(void)
{
    x = 10;
    do_something();
    assert(x == 11);  /* x was havoced, no ensures to constrain it */
    return 0;
}
