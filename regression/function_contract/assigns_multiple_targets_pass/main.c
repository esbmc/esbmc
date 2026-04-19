/* Test: __ESBMC_assigns with multiple targets
 * 
 * Verifies that __ESBMC_assigns(a, b, c) correctly havocs
 * only the specified targets, leaving others unchanged.
 */
#include <assert.h>

int a, b, c, d, e;

void modify_abc() {
    __ESBMC_assigns(a, b, c);
    __ESBMC_ensures(a == 1);
    __ESBMC_ensures(b == 2);
    __ESBMC_ensures(c == 3);
    
    a = 1;
    b = 2;
    c = 3;
}

int main() {
    a = 10;
    b = 20;
    c = 30;
    d = 40;
    e = 50;
    
    modify_abc();
    
    // a, b, c modified as per ensures
    assert(a == 1);
    assert(b == 2);
    assert(c == 3);
    
    // d, e should be unchanged (not in assigns)
    assert(d == 40);
    assert(e == 50);
    
    return 0;
}
