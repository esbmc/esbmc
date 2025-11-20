// test_const_prop_struct.c
#include <assert.h>

struct Point {
    int x;
    int y;
};

int test_struct_field_arithmetic() {
    struct Point p;
    p.x = 10;
    p.y = 20;
    int result = p.x + p.y;  // Should propagate: result = 30
    return result;
}

int test_struct_field_comparison() {
    struct Point p1, p2;
    p1.x = 5;
    p1.y = 10;
    p2.x = 5;
    p2.y = 10;
    int result = (p1.x == p2.x) && (p1.y == p2.y);  // Should propagate: result = 1
    return result;
}

struct Vector {
    float x;
    float y;
    float z;
};

float test_struct_float_operations() {
    struct Vector v;
    v.x = 1.0f;
    v.y = 2.0f;
    v.z = 3.0f;
    float result = v.x + v.y + v.z;  // Should propagate: result = 6.0f
    return result;
}

int main() {
    assert(test_struct_field_arithmetic() == 30);
    assert(test_struct_field_comparison() == 1);
    assert(test_struct_float_operations() == 6.0f);
    return 0;
}
