#include <assert.h>

int main() {
    char buffer[100];
    char *ptr1 = buffer;
    char *ptr2 = buffer + 50;
    
    // Both should return 100 (full array size)
    assert(__builtin_object_size(ptr1, 0) == 100);
    assert(__builtin_object_size(ptr2, 0) == 100);
    
    return 0;
}

