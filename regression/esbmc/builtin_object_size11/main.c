#include <assert.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    char buffer[100];
    char *ptr = buffer + argc; // symbolic offset
    size_t size = __builtin_object_size(ptr, 1);
    assert(size == (size_t)-1);
    return 0;
}

