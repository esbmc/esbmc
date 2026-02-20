/* Test: __ESBMC_assigns(arr) havocs entire array - FAIL expected
 * 
 * When assigns clause specifies entire array, ALL elements are havoc'd.
 * If ensures clause doesn't constrain an element, verification should fail.
 */
#include <assert.h>

int arr[3];

void modify_array() {
    __ESBMC_assigns(arr);           // Havocs entire array
    __ESBMC_ensures(arr[0] == 1);   // Only constrains arr[0]
    // BUG: arr[1] and arr[2] are havoc'd but not constrained!
    arr[0] = 1;
    arr[1] = 2;
    arr[2] = 3;
}

int main() {
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    
    modify_array();
    
    assert(arr[0] == 1);   // Should pass - constrained by ensures
    assert(arr[1] == 2);   // Should FAIL - arr[1] was havoc'd, not constrained
    
    return 0;
}
