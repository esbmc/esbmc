#include <stdlib.h>
#include <assert.h>

int test_realloc_failure() {
    int *ptr = malloc(5 * sizeof(int));
    assert(ptr != NULL);
    
    ptr[0] = 123;
    
    // Try to allocate (might fail)
    int *new_ptr = realloc(ptr, 10);
    
    if (new_ptr == NULL) {
        // Realloc failed, original pointer is still valid
        assert(ptr[0] == 123);
        free(ptr);  // Free original memory
    } else {
        // Realloc succeeded
        assert(new_ptr[0] == 123);
        free(new_ptr);
    }
    
    return 0;
}

int main() {
  test_realloc_failure();
  return 0;
}
