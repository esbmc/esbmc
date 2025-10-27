#include <stdlib.h>
#include <assert.h>

int test_realloc_null() {
    // realloc(NULL, size) is equivalent to malloc(size)
    int *ptr = realloc(NULL, 5 * sizeof(int));
    assert(ptr != NULL);
    
    for(int i = 0; i < 5; i++) {
        ptr[i] = i * 2;
    }
    
    assert(ptr[2] == 4);
    
    free(ptr);
    return 0;
}

int main() {
  test_realloc_null();
  return 0;
}
