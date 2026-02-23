#include <assert.h>

int main() {
    char buffer[100];
    char *ptr1 = buffer;
    char *ptr2 = buffer + 50;
    assert(__builtin_object_size(ptr1, 0) == 10);
    assert(__builtin_object_size(ptr2, 0) == 1000);
    return 0;
}

