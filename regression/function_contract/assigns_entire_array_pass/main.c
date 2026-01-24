/* Test: __ESBMC_assigns with entire array
 * 
 * Verifies that __ESBMC_assigns(arr) havocs the entire array.
 * When assigning entire array, postcondition must constrain
 * all elements that need to be verified.
 */
#include <assert.h>

int arr[3];

void init_array() {
    __ESBMC_assigns(arr);
    __ESBMC_ensures(arr[0] == 0);
    __ESBMC_ensures(arr[1] == 1);
    __ESBMC_ensures(arr[2] == 2);
    
    arr[0] = 0;
    arr[1] = 1;
    arr[2] = 2;
}

int main() {
    init_array();
    
    assert(arr[0] == 0);
    assert(arr[1] == 1);
    assert(arr[2] == 2);
    
    return 0;
}
