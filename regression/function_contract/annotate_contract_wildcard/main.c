/* Test: Contract annotation with wildcard --enforce-contract *
 * 
 * This test verifies that --enforce-contract * correctly picks up
 * all functions marked with the contract annotation.
 */
#include <assert.h>

int counter = 0;
int value = 0;

__attribute__((annotate("__ESBMC_contract")))
void increment(void)
{
    __ESBMC_requires(counter >= 0);
    __ESBMC_ensures(counter == __ESBMC_old(counter) + 1);
    __ESBMC_assigns(counter);
    
    counter++;
}

__attribute__((annotate("__ESBMC_contract")))
void set_value(int v)
{
    __ESBMC_requires(v >= 0);
    __ESBMC_ensures(value == v);
    __ESBMC_assigns(value);
    
    value = v;
}

int main(void)
{
    counter = 5;
    value = 0;
    
    increment();
    set_value(42);
    
    assert(counter == 6);
    assert(value == 42);
    
    return 0;
}
