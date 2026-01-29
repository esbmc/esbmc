/* Test: auto-havoc with function that doesn't affect main's assertions
 * 
 * This test demonstrates that when a function marked with #pragma contract
 * doesn't affect the variables checked in main's assertions, the
 * conservative havoc won't cause false failures.
 * 
 * The function useless_computation() only modifies its local variable y,
 * which doesn't escape the function scope. The main function's assertion
 * on x should still pass even after havoc, because:
 * 1. useless_computation() has no side effects on x
 * 2. Conservative havoc may havoc globals, but x is set after the call
 * 3. The assertion should pass
 */
#include <assert.h>

int y = 0;

#pragma contract
void useless_computation(void)
{
    int local = y + 1;
    y = local * 2;
}

int main()
{
    int x = 10;
    
    useless_computation();
    
    // x is local and set after the call, so it's not affected
    // This should pass even with conservative havoc
    assert(x == 10);
    
    return 0;
}
