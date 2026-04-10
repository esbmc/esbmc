/* Test: __ESBMC_assigns(x) havocs scalar - FAIL expected
 * 
 * When assigns clause specifies a scalar, it is havoc'd.
 * If ensures clause doesn't constrain it, verification should fail.
 */
#include <assert.h>

int global_x;
int global_y;

void modify_x() {
    __ESBMC_assigns(global_x);  // Havocs global_x
    __ESBMC_ensures(1);         // Trivial ensures - doesn't constrain global_x!
    global_x = 42;
}

int main() {
    global_x = 10;
    global_y = 20;
    
    modify_x();
    
    // global_y should be unchanged (not in assigns)
    assert(global_y == 20);
    
    // global_x was havoc'd but not constrained - should FAIL
    assert(global_x == 42);
    
    return 0;
}
