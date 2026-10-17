#include <assert.h>

int main() {
    char buffer[100];
    char *ptr1 = buffer;
    char *ptr2 = buffer + 50;
    
    // Type 0 counts from the pointer to the end of the array (#8660)
    assert(__builtin_object_size(ptr1, 0) == 100);
    assert(__builtin_object_size(ptr2, 0) == 50);
    
    return 0;
}

