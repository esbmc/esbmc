#include <stdlib.h>
#include <assert.h>

int main() {
    // Test Case: Buffer overflow after shrinking - should fail
    int *ptr = malloc(10 * sizeof(int));
    assert(ptr != NULL);
    
    // Initialize all 10 elements
    for(int i = 0; i < 10; i++) {
        ptr[i] = i * 10;
    }
    
    // Shrink memory to only 5 elements
    ptr = realloc(ptr, 5 * sizeof(int));
    assert(ptr != NULL);
    
    // This should still work - within bounds
    assert(ptr[0] == 0);
    assert(ptr[4] == 40);
    
    // BUG: Access beyond new allocated size!
    ptr[7] = 999;  // Buffer overflow - only 5 elements allocated now
    
    free(ptr);
    return 0;
}
