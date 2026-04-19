#include <assert.h>

int main() {
    int i = 0;
    int j = 0;
    int k = 0;
    int sum = 0;
    
    // Mixed case: outer and innermost loops have invariants, middle loop doesn't
    __ESBMC_loop_invariant(i >= 0 && i <= 2);
    while (i < 2) {
        j = 0;
        
        // Middle loop has no invariant
        while (j < 3) {
            k = 0;
            
            __ESBMC_loop_invariant(k >= 0 && k <= 4);
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
