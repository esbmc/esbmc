#include <assert.h>

int main() {
    int i = 0;
    int j = 0;
    int k = 0;
    int sum = 0;
    
    // Triple nested loops with invariants
    __ESBMC_loop_invariant(i >= 0 && i <= 2);
    __ESBMC_loop_invariant(sum == i * 12);
    while (i < 2) {
        j = 0;
        
        __ESBMC_loop_invariant(j >= 0 && j <= 3);
        __ESBMC_loop_invariant(sum == i * 12 + j * 4);
        while (j < 3) {
            k = 0;
            
            __ESBMC_loop_invariant(k >= 0 && k <= 4);
            __ESBMC_loop_invariant(sum == i * 12 + j * 4 + k);
            while (k < 4) {
                sum++;
                k++;
            }
            j++;
        }
        i++;
    }
    
    assert(sum == 24); // 2 * 3 * 4 = 24
    return 0;
}
