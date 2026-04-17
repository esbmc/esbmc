/* Test that --enforce-contract detects ensures clause violations with __ESBMC_assigns
 * 
 * This test verifies that when a function violates its ensures clause,
 * the --enforce-contract mode properly detects the violation even when
 * the function has __ESBMC_assigns clauses.
 * 
 * This ensures the fix for assigns_target handling doesn't break
 * the contract verification logic.
 */
#include <assert.h>
#include <stddef.h>

int global_x = 0;

int increment_and_return()
{
    __ESBMC_assigns(global_x);
    __ESBMC_ensures(global_x == __ESBMC_old(global_x) + 1);
    __ESBMC_ensures(__ESBMC_return_value == global_x);
    
    // BUG: Violates ensures - global_x not incremented correctly
    global_x = global_x + 2;  // Should be +1, not +2!
    return global_x;
}

int main()
{
    global_x = 10;
    int result = increment_and_return();
    
    return 0;
}
