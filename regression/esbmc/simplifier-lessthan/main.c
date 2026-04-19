#include <assert.h>
#include <stdio.h>

void test_union_updates() {
    union Data {
        int i;
        float f;
        char c;
    };
    
    union Data data;
    
    // Update same union field multiple times
    data.i = 100;
    data.i = 200;
    data.i = 300;
    
    assert(data.i == 300);
}

int main() {
    test_union_updates();
    
    return 0;
}
