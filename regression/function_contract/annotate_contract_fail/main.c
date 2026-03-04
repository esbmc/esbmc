/* Test: Contract annotation with postcondition violation
 * 
 * This test verifies that contract violations are detected correctly.
 * The function claims to increment by 1, but actually increments by 2.
 */
#include <assert.h>

int counter = 0;

__ESBMC_contract
void increment(void)
{
    __ESBMC_requires(counter >= 0);
    // BUG: says increment by 1, but actually increments by 2
    __ESBMC_ensures(counter == __ESBMC_old(counter) + 1);
    __ESBMC_assigns(counter);
    
    counter = counter + 2;  // Violates postcondition!
}

int main(void)
{
    counter = 10;
    increment();
    return 0;
}
