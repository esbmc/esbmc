#include <assert.h>

int main() {
    int x = 42;
    char c = 'a';
    double d = 3.14;
    
    assert(__builtin_object_size(&x, 0) == sizeof(char));
    assert(__builtin_object_size(&c, 0) == sizeof(int));
    assert(__builtin_object_size(&d, 0) == sizeof(float));
    
    return 0;
}

