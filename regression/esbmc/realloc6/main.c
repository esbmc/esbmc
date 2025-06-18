#include <stdlib.h>
#include <assert.h>

int test_chained_reallocs() {
    int *ptr = malloc(sizeof(int));
    assert(ptr != NULL);
    
    *ptr = 42;
    
    // Chain of reallocs
    ptr = realloc(ptr, 2 * sizeof(int));
    assert(ptr != NULL && ptr[0] == 42);
    
    ptr = realloc(ptr, 4 * sizeof(int));
    assert(ptr != NULL && ptr[0] == 42);
    
    ptr = realloc(ptr, 8 * sizeof(int));
    assert(ptr != NULL && ptr[0] == 42);
    
    // Shrink back down
    ptr = realloc(ptr, 2 * sizeof(int));
    assert(ptr != NULL && ptr[0] == 42);
    
    free(ptr);
    return 0;
}

int main() {
  test_chained_reallocs();
  return 0;
}
