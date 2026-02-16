#include <assert.h>
#include <stdlib.h>

int main() {
    char buffer[50];
    size_t size = __builtin_object_size(buffer, 0);
    assert(size == 50);
    assert(size > 0);
    assert(size == (size_t)-1);
    return 0;
}

