/* Test: Contract annotation with NO contract specification at all
 * 
 * This test verifies that a function marked with the annotation
 * but without any __ESBMC_requires, __ESBMC_ensures, or __ESBMC_assigns
 * uses default contracts (requires(true), ensures(true)).
 * 
 * In enforce mode, this should always pass since:
 * - requires(true) is trivially satisfied
 * - ensures(true) is trivially satisfied
 */
#include <assert.h>

int x = 0;

__ESBMC_contract
void do_something(void)
{
    // No contract specification at all
    // Will use default: requires(true), ensures(true)
    x = x + 1;
}

int main(void)
{
    x = 10;
    do_something();
    // We cannot assert x == 11 because we don't have ensures clause
    // But the function should still be processed
    return 0;
}
