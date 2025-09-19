#include <assert.h>

int main() {
    int x = nondet_int();
    unsigned u = nondet_uint();
    
    int zero_int = 0;
    assert((x ^ zero_int) == x);
    
    int zero_for_comparison = 0;
    assert((u < zero_for_comparison) == 0);
    
    assert((x ^ 0) == x);
    assert((u < 0) == 0);
    
    return 0;
}
