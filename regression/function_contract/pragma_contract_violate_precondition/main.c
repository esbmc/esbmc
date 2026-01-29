/* Test: Conservative havoc makes verification fail on correct code
 * 
 * This test demonstrates over-approximation: the function is correct
 * and would normally pass, but because it uses default conservative contract
 * (no explicit ensures), the havoc makes x unpredictable.
 */
#include <assert.h>

int x = 0;

// No explicit contract - will get default require(true), ensure(true)
#pragma contract
void increment(void)
{
    x = x + 1;
}

int main(void)
{
    x = 5;
    
    increment();
    
    // With conservative havoc, x can be anything
    // So this assertion fails even though the code is correct
    assert(x == 6);
    
    return 0;
}
