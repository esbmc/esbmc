/* Test: #pragma contract with wrong explicit contract - should fail
 * 
 * This test demonstrates that even with #pragma contract, if the explicit
 * contract is wrong (doesn't match the implementation), verification will fail.
 * 
 * The contract claims the function increments by 1, but it actually increments by 2.
 * This should be caught during contract enforcement.
 */
#include <assert.h>

int counter = 0;

#pragma contract
void increment_by_two(void)
{
    __ESBMC_requires(counter >= 0);
    __ESBMC_requires(counter < 1000);
    // WRONG: says increment by 1, but actually increments by 2
    __ESBMC_ensures(counter == __ESBMC_old(counter) + 1);
    __ESBMC_assigns(counter);
    
    counter++;
    counter++;  //added an error here
}

int main(void)
{
    counter = 10;
    
    // This call should fail during contract enforcement
    increment_by_two();
    
    return 0;
}
