/* Minimal test: function depends on global array state
 * 
 * KNOWNBUG: In --replace-call-with-contract mode, functions that read global
 * arrays cannot see values modified by previous calls. Requires dynamic frame
 * conditions to track actual modifications.
 */
#include <assert.h>

int arr[3];

void set_arr(int i) {
    __ESBMC_assigns(arr);
    __ESBMC_ensures(arr[i] == 1);
    arr[i] = 1;
}

void check_arr() {
    __ESBMC_assigns(0);
    assert(arr[0] == 1);  // Fails: cannot see arr[0] modified by set_arr(0)
}

int main() {
    set_arr(0);
    check_arr();
    return 0;
}

