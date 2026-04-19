#include <assert.h>

union Data {
    int i;
    char str[20];
};

union Log {
    int x;
    char str[30];
};

int main() {
    union Data data;
    assert(__builtin_object_size(&data, 0) == sizeof(union Log));
    return 0;
}

