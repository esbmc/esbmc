#include <stdlib.h>
#include <stddef.h>

struct small_struct {
    char data[4];
};

int main() {
    struct small_struct *ptr = malloc(sizeof *ptr);
    
    // Try to access way beyond the allocated memory
    char *base = (char*)ptr;
    char *far_ptr = base + 1000;  // Way out of bounds
    
    // This should fail - accessing unallocated memory
    char bad_access = *far_ptr;
    
    free(ptr);
    return 0;
}
