/* Minimal test: assigns entire array but only modifies one element
 * 
 * KNOWNBUG: In --replace-call-with-contract mode, assigning entire array
 * in assigns clause causes all elements to be havoc'd, even when only one
 * element is modified. Requires dynamic frame conditions.
 */
#include <assert.h>

int arr[3];

void modify_one(int i) {
    __ESBMC_assigns(arr);  // Assigns entire array
    __ESBMC_ensures(arr[i] == 2);
    arr[i] = 2;  // Only modifies arr[i]
}

int main() {
    arr[0] = 1;
    modify_one(1);  // Only modifies arr[1]
    assert(arr[0] == 1);  // Fails: arr[0] was havoc'd
    return 0;
}

