/* Test: precise array element assigns
 * 
 * This test verifies that __ESBMC_assigns(arr[i]) only havocs the
 * specified element, leaving other elements unchanged.
 */
#include <assert.h>

int arr[3];

void modify_one(int i) {
    __ESBMC_requires(i >= 0 && i < 3);
    __ESBMC_assigns(arr[i]);  // Only assigns arr[i]
    __ESBMC_ensures(arr[i] == 2);
    arr[i] = 2;  // Only modifies arr[i]
}

int main() {
    arr[0] = 1;
    modify_one(1);  // Only modifies arr[1]
    assert(arr[0] == 1);  // Passes: arr[0] unchanged
    return 0;
}

