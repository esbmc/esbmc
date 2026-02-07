/* Test: __ESBMC_assigns with array element - precise havoc
 * 
 * Verifies that __ESBMC_assigns(arr[i]) only havocs arr[i],
 * leaving other elements unchanged.
 */
#include <assert.h>

int arr[5];

void set_element(int i, int val) {
    __ESBMC_requires(i >= 0 && i < 5);
    __ESBMC_assigns(arr[i]);
    __ESBMC_ensures(arr[i] == val);
    arr[i] = val;
}

int main() {
    // Initialize array
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    arr[3] = 40;
    arr[4] = 50;
    
    // Modify only arr[2]
    set_element(2, 99);
    
    // Check: arr[2] was modified, others unchanged
    assert(arr[0] == 10);
    assert(arr[1] == 20);
    assert(arr[2] == 99);
    assert(arr[3] == 40);
    assert(arr[4] == 50);
    
    return 0;
}
