/* Test: #pragma contract with postcondition failure
 * 
 * This test demonstrates that a function implementation that doesn't
 * satisfy its postcondition will be caught during contract enforcement.
 * 
 * The function claims to set x to 100, but actually sets it to 99.
 */
#include <assert.h>

int x = 0;

#pragma contract
void set_to_hundred(void)
{
    __ESBMC_ensures(x == 100);
    __ESBMC_assigns(x);
    
    x = 99;  // BUG: should be 100!
}

int main(void)
{
    x = 0;
    
    // This should fail during contract enforcement
    // because the implementation violates the postcondition
    set_to_hundred();
    
    return 0;
}
