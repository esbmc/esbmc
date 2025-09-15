#include <assert.h>

struct Point {
    int x, y;
};

int main() {
    struct Point p;
    assert(__builtin_object_size(&p, 0) == sizeof(struct Point));
    return 0;
}

