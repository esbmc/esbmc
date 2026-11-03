#include <assert.h>

struct Inner {
    int values[5];
};

struct Outer {
    struct Inner inner;
    char padding[10];
};

int main() {
    struct Outer outer;
    
    assert(__builtin_object_size(&outer, 0) == sizeof(struct Outer));
    // Type 0 measures the whole object, not the member (#8660)
    assert(__builtin_object_size(&outer.inner, 0) == sizeof(struct Outer));
    
    return 0;
}

