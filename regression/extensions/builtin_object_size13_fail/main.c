#include <assert.h>
#include <stdlib.h>

union U {
    char c;
    long long big; // may introduce padding/alignment
};

int main() {
    union U u;
    size_t size = __builtin_object_size(&u, 0);

    assert(size == sizeof(long long) - 1);
    return 0;
}

