#include <assert.h>

char global_buffer[64];

int main() {
    assert(__builtin_object_size(global_buffer, 0) == 128);
    return 0;
}

