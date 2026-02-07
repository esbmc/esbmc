/* Test: __ESBMC_assigns with expressions - PASS case
 * Expected: VERIFICATION SUCCESSFUL
 * Tests that assigns clause correctly havoc only specified expressions
 * and allows verification to succeed when contract is satisfied
 */
#include <assert.h>

int global_x = 0;
int global_y = 0;

void modify_x(int *ptr) {
    __ESBMC_assigns(global_x, *ptr);
    __ESBMC_ensures(global_x == 42);
    __ESBMC_ensures(*ptr == 100);
    
    global_x = 42;
    *ptr = 100;
}

int main() {
    int local = 0;
    global_x = 0;
    global_y = 200;
    
    // Call function with contract in replace-call mode
    modify_x(&local);
    
    // After replace-call:
    // - global_x and local should be havoc'd, but ensures says they have specific values
    // - global_y should NOT be havoc'd (not in assigns clause) and should remain 200
    assert(global_x == 42);
    assert(local == 100);
    assert(global_y == 200);  // Should remain unchanged (not in assigns clause)
    
    return 0;
}

