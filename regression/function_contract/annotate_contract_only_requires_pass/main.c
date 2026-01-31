/* Test: Contract annotation with only requires (no ensures, no assigns)
 * 
 * This test verifies that a function with only precondition works correctly.
 * ensures defaults to true, assigns defaults to conservative havoc.
 */
#include <assert.h>

int value = 0;

__attribute__((annotate("__ESBMC_contract")))
void set_positive(int x)
{
    __ESBMC_requires(x > 0);
    // No ensures, no assigns
    // ensures defaults to true
    // assigns defaults to conservative havoc of all globals
    
    value = x;
}

int main(void)
{
    set_positive(42);  // Valid call: 42 > 0
    return 0;
}
