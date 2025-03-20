#include <assert.h>
#include <stdio.h>

int main() {
    // Test AND operation
    assert((1 & 1) == 1); // 1 AND 1 should be 1
    assert((1 & 0) == 0); // 1 AND 0 should be 0
    assert((0 & 1) == 0); // 0 AND 1 should be 0
    assert((0 & 0) == 0); // 0 AND 0 should be 0
    
    printf("AND operation tests passed.\n");
    return 0;
}
