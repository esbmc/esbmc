/* Test: #pragma contract with explicit contract - should pass
 * 
 * This test demonstrates using #pragma contract with explicit contract
 * specifications. The function has precise preconditions, postconditions,
 * and assigns clauses that accurately describe its behavior.
 * 
 * With precise contracts, verification should succeed because:
 * 1. The contract accurately describes the function behavior
 * 2. No over-approximation is needed
 * 3. The main function's assertions are provable from the contract
 */
#include <assert.h>

int counter = 0;

#pragma contract
void increment_by_two(void)
{
    __ESBMC_requires(counter >= 0);
    __ESBMC_requires(counter < 1000);
    __ESBMC_ensures(counter == __ESBMC_old(counter) + 2);
    __ESBMC_assigns(counter);
    
    counter++;
    counter++;
}

int main(void)
{
    counter = 10;
    
    // Call the function with precise contract
    increment_by_two();
    
    // With precise contract, we can prove this
    assert(counter == 12);
    
    increment_by_two();
    
    // And this too
    assert(counter == 14);
    
    return 0;
}
