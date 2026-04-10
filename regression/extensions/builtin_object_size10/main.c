#include <assert.h>
#include <stdlib.h>

int main() {
    char *buf = (char *)malloc(20);
    size_t size = __builtin_object_size(buf, 0);
    assert(size == 20);
    free(buf);
    return 0;
}

