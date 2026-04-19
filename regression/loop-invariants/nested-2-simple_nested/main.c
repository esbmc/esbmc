#include <assert.h>

int main() {
    int i = 0;
    int j = 0;
    int sum = 0;
    
    // Simple nested loops with invariants
    __ESBMC_loop_invariant(i >= 0 && i <= 2);
    while (i < 2) {
        j = 0;
        
        __ESBMC_loop_invariant(j >= 0 && j <= 3);
        while (j < 3) {
            sum++;
            j++;
        }
        i++;
    }
    
    assert(sum == 6); // 2 * 3 = 6
    return 0;
}
