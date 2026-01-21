/* Test __ESBMC_old with logical operators (&&, ||)
 * This tests if __ESBMC_old can be used within && and || expressions
 */
#include <stddef.h>

// Test case 1: __ESBMC_old with || (using simple int parameter)
void test_or(int *x) {
    __ESBMC_requires(x != NULL);
    
    // If old value was 0, then new value is 1
    __ESBMC_ensures(__ESBMC_old(*x) != 0 || *x == 1);
    
    if (*x == 0) {
        *x = 1;
    }
}

// Test case 2: __ESBMC_old with &&
void test_and(int *x) {
    __ESBMC_requires(x != NULL);
    
    // Old value was 0 AND new value should be 1
    __ESBMC_ensures(__ESBMC_old(*x) == 0 && *x == 1);
    
    *x = 1;
}

// Test case 3: Multiple __ESBMC_old in one expression
void test_multiple_old(int *x) {
    __ESBMC_requires(x != NULL);
    
    // Both old and new value should be non-zero
    __ESBMC_ensures(__ESBMC_old(*x) == 0 || (__ESBMC_old(*x) != 0 && *x != 0));
    
    if (*x == 0) {
        *x = 1;
    }
}

int main() {
    int x1 = 0;
    test_or(&x1);
    
    int x2 = 0;
    test_and(&x2);
    
    int x3 = 5;
    test_multiple_old(&x3);
    
    return 0;
}
