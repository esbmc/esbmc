#include <assert.h>

int main() {
    int i = 0;
    int j = 0;
    int sum = 0;
    
    // Outer loop with multiple invariants
    __ESBMC_loop_invariant(i >= 0);
    __ESBMC_loop_invariant(i <= 2);
    __ESBMC_loop_invariant(sum == i * 3);
    while (i < 2) {
        j = 0;
        
        // Inner loop with multiple invariants
        __ESBMC_loop_invariant(j >= 0);
        __ESBMC_loop_invariant(j <= 3);
        __ESBMC_loop_invariant(sum == i * 3 + j);
        while (j < 3) {
            sum++;
            j++;
        }
        i++;
    }
    
    assert(sum == 6); // 2 * 3 = 6
    return 0;
}
