/* Test: Contract annotation with default contract (only assigns)
 * 
 * This test verifies that functions marked with annotation but without
 * explicit contracts will use default contracts (requires(true), ensures(true)).
 * 
 * The function only has an assigns clause, which limits havoc scope.
 */
#include <assert.h>

int value = 0;
int other = 100;

__ESBMC_contract
void modify_value(void)
{
    __ESBMC_assigns(value);  // Only 'value' is modified
    
    value = value + 5;
}

int main(void)
{
    value = 10;
    other = 100;
    
    modify_value();
    
    // With assigns(value), 'other' is NOT havoced in replace mode
    assert(other == 100);  // This should pass
    
    return 0;
}
