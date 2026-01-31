/* Test: __attribute__((annotate("__ESBMC_contract"))) with explicit contract
 * 
 * This test verifies that the contract annotation correctly marks the
 * function for contract processing.
 */
#include <assert.h>

int counter = 0;

__attribute__((annotate("__ESBMC_contract")))
void increment(void)
{
    __ESBMC_requires(counter >= 0);
    __ESBMC_requires(counter < 1000);
    __ESBMC_ensures(counter == __ESBMC_old(counter) + 1);
    __ESBMC_assigns(counter);
    
    counter++;
}

int main(void)
{
    counter = 10;
    increment();
    assert(counter == 11);
    return 0;
}
