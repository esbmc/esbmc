/* Test: auto-havoc with conservative havoc causes over-approximation
 * 
 * This test demonstrates that the conservative default contract
 * (require(true), ensure(true)) with havoc produces over-approximation,
 * which may lead to false counterexamples.
 * 
 * The function increment() always increments x by 1, so after calling it
 * twice, x should be 12. However, with conservative havoc:
 * - require(true): no precondition check
 * - havoc all globals (including x)
 * - ensure(true): no postcondition constraint
 * 
 * This allows x to have ANY value after the call, so the assertion
 * may fail, demonstrating over-approximation.
 */
#include <assert.h>

int x = 10;

#pragma contract
void increment(void)
{
    x = x + 1;
}

int main()
{
    increment();
    increment();
    
    // This should pass in reality (x == 12)
    // But with conservative havoc, x can be anything
    // So this may fail, showing over-approximation
    assert(x == 12);
    
    return 0;
}
