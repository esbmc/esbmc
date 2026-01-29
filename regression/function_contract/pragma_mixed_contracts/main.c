/* Test: #pragma contract with mixed explicit and default contracts
 * 
 * This test demonstrates mixing functions with explicit contracts
 * and functions with default contracts in the same program.
 * 
 * - add_one: has explicit precise contract
 * - multiply_by_unknown: has default conservative contract
 * 
 * This shows that auto-havoc can handle both cases intelligently.
 */
#include <assert.h>

int value = 0;
int other = 0;

// Function with explicit precise contract
#pragma contract
void add_one(void)
{
    __ESBMC_requires(value < 100);
    __ESBMC_assigns(value);
    __ESBMC_ensures(value == __ESBMC_old(value) + 1);
    
    value = value + 1;
}

// Function with default conservative contract (no explicit contract)
#pragma contract
void multiply_by_unknown(void)
{
    // No explicit contract, will get require(true), ensure(true)
    // This function modifies 'other' in an unknown way
    other = other * 2 + 3;
}

int main(void)
{
    value = 10;
    other = 5;
    
    // Call function with precise contract
    add_one();
    
    // With precise contract, we can prove this
    assert(value == 11);
    
    add_one();
    assert(value == 12);
    
    // Call function with default conservative contract
    multiply_by_unknown();
    
    // WARNING: Conservative havoc affects ALL globals
    // So we cannot prove anything about 'value' or 'other' after multiply_by_unknown()
    // Even though multiply_by_unknown() doesn't touch 'value' in reality,
    // the conservative havoc doesn't know that
    
    // The only thing we can prove is about local variables set AFTER the call
    int x = 100;
    assert(x == 100);  // This is safe
    
    return 0;
}
