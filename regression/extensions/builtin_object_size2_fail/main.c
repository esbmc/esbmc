#include <assert.h>

struct Point {
    int x, y;
};

struct Line {
    int z;
};

int main() {
    struct Point p;
    assert(__builtin_object_size(&p, 0) == sizeof(struct Line));
    return 0;
}

