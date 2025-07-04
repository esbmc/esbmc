#include <stdlib.h>
#include <assert.h>

int main() {
    // Test Case: Correct realloc usage - should pass
    int *ptr = malloc(5 * sizeof(int));
    assert(ptr != NULL);
    
    // Initialize memory
    for(int i = 0; i < 5; i++) {
        ptr[i] = i;
    }
    
    // Expand memory
    ptr = realloc(ptr, 10 * sizeof(int));
    assert(ptr != NULL);
    
    // Verify data preservation
    assert(ptr[0] == 0);
    assert(ptr[4] == 4);
    
    // Use new memory
    for(int i = 5; i < 10; i++) {
        ptr[i] = i;
    }
    
    // Verify all data
    assert(ptr[9] == 9);
    
    free(ptr);
    return 0;
}
