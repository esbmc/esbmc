/* Test: enforce-all-contracts with no contract specification, assert fails
 *
 * Function is annotated but has no requires/ensures/assigns.
 * In enforce mode, requires(true) and ensures(true) are used as defaults.
 * The function body still executes normally, so the bug (adding 2
 * instead of 1) is caught by the assert in main.
 */
#include <assert.h>

int x = 0;

__attribute__((annotate("__ESBMC_contract")))
void increment(void)
{
    /* Bug: adds 2 instead of 1 */
    x = x + 2;
}

int main(void)
{
    x = 10;
    increment();
    assert(x == 11);  /* Fails: x is actually 12 */
    return 0;
}
