#include <assert.h>

union Data {
    int i;
    char str[20];
};

int main() {
    union Data data;
    assert(__builtin_object_size(&data, 0) == sizeof(union Data));
    return 0;
}

