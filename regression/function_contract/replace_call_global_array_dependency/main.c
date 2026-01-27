/* Test: function depends on global array state with precise assigns
 * 
 * This test verifies that when using precise __ESBMC_assigns for
 * array elements, subsequent functions can see the modifications
 * through the ensures clause.
 */
#include <assert.h>

int arr[3];

void set_arr(int i) {
    __ESBMC_requires(i >= 0 && i < 3);
    __ESBMC_assigns(arr[i]);  // Precise: only assigns arr[i]
    __ESBMC_ensures(arr[i] == 1);
    arr[i] = 1;
}

void check_arr() {
    __ESBMC_assigns();  // Pure function: no side effects
    assert(arr[0] == 1);  // Passes: arr[0] was set by set_arr(0)
}

int main() {
    set_arr(0);
    check_arr();
    return 0;
}

