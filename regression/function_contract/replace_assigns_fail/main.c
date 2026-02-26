/* Test: __ESBMC_assigns with expressions - FAIL case
 * Expected: VERIFICATION FAILED
 * Tests that assigns clause correctly preserves non-assigned variables,
 * and verification fails when we assert a contradiction
 * 
 * In replace-call mode:
 * - global_x is in assigns, so it's havoc'd then assumed to be 42 (from ensures)
 * - global_y is NOT in assigns, so it remains unchanged (0)
 * - We assert global_y == 100, which contradicts the fact that it should remain 0
 */
#include <assert.h>

int global_x = 0;
int global_y = 0;

void modify_x() {
    __ESBMC_assigns(global_x);
    // Only global_x is in assigns clause, not global_y
    __ESBMC_ensures(global_x == 42);
    
    global_x = 42;
    // Note: In replace-call mode, function body is not executed
    // Only the contract (assigns + ensures) is used
}

int main() {
    global_x = 0;
    global_y = 0;
    
    // Call function with contract in replace-call mode
    modify_x();
    
    // After replace-call:
    // - global_x is havoc'd, then assumed to be 42 (from ensures clause)
    // - global_y is NOT havoc'd (not in assigns clause), so it remains 0
    assert(global_x == 42);  // This should pass (ensures clause)
    
    // This assertion contradicts the fact that global_y should remain 0
    // (since it's not in assigns clause), so verification should FAIL
    assert(global_y == 100);  // This should fail - global_y should be 0, not 100
    
    return 0;
}

