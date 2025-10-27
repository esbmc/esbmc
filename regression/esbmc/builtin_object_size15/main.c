#include <assert.h>

char global_buffer[64];

int main() {
    assert(__builtin_object_size(global_buffer, 0) == 64);
    return 0;
}

