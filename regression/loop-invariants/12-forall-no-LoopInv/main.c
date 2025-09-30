/*
Current work around with ESBMC quantifier without loop invariant checker
*/
#include <assert.h>

extern int nondet_int();
extern _Bool __ESBMC_forall(void *, _Bool);

int max(int *arr, unsigned n) {
    int max_val = arr[0];
    unsigned i = 1;
    
    // Define unified loop invariant
    unsigned j;
    #define loop_invariant (i >= 1 && i <= n && __ESBMC_forall(&j, !(j >= 0 && j < i) || (max_val >= arr[j])))
    
    // 1. Assert invariants before entering the loop (initial state)
    __ESBMC_assert(loop_invariant, "Initial: loop invariant should hold");
    
    // 2. Capture all related variables in the loop (simulate step k)
    max_val = nondet_int();
    i = nondet_int();
    
    // 3. Set the loop invariant as the assumption (at step k)
    __ESBMC_assume(loop_invariant);
    
    // 4. Enter the loop (only run a single step of the loop)
    // Branch 1: loop continues
    if (i < n) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
        i = i + 1;
        
        // 5. Check if the invariant is satisfiable after the loop body (step k+1)
        __ESBMC_assert(loop_invariant, "After iteration: loop invariant should hold");
        
        // 6. Terminate the loop
        __ESBMC_assume(0);
    }
    // Branch 2: loop exits
    else {
        // At this point we have:
        // - invariant: max_val >= arr[j] for all 0 <= j < i
        // - exit condition: !(i < n), which means i >= n
        // - from invariant bound: i <= n
        // Therefore: i == n
        
        // This means max_val >= arr[j] for all 0 <= j < n
        // So the postcondition holds
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
    
    // Verify postcondition: result is greater than or equal to all elements
    unsigned i;
    _Bool postcondition = __ESBMC_forall(&i, 
        !(i >= 0 && i < n) || (result >= a[i]));
    assert(postcondition);
    
    return 0;
}