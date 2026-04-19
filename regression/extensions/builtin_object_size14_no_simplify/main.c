#include <assert.h>
#include <stdlib.h>

int main() {
    char *p;
    // Uninitialized pointer, object size cannot be determined
    size_t size = __builtin_object_size(p, 0);

    assert(size == (size_t)-1);
    return 0;
}

