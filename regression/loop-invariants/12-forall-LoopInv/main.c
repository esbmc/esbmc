#include <assert.h>

extern _Bool __ESBMC_forall(void *, _Bool);

int max(int *arr, unsigned n) {

    
    int max_val = arr[0];
    unsigned i = 1;
    
    unsigned j;
    __ESBMC_loop_invariant(
        i >= 1 && i <= n &&
        __ESBMC_forall(&j, !(j >= 0 && j < i) || (max_val <= arr[j]))
    );
    
    for (; i < n; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    
    return max_val;
}

int main() {
    unsigned n;
    __ESBMC_assume(n > 0 && n <= 20);
    
    int a[n];
    
    // Initialize array with bounded values
    for (unsigned i = 0; i < n; i++) {
        __ESBMC_assume(a[i] >= -1000 && a[i] <= 1000);
    }
    
    int result = max(a, n);
    
    // Verify postcondition: result >= all elements
    unsigned i;
    _Bool postcondition = __ESBMC_forall(&i, 
        !(i >= 0 && i < n) || (result >= a[i]));
    assert(postcondition);
    
    return 0;
}