#include <assert.h>

struct Counter {
    int *ptr;
    
    Counter(int &r) : ptr(&r) {}
    void increment() { (*ptr)++; }
    void decrement() { (*ptr)--; }
    void decrement2() { (*ptr -= 1);}
    void decrement3() { --(*ptr);}
};

int main() {
    int x = 5;
    Counter c(x);
    
    // Test single increment
    c.increment();
    assert(x == 6);  // x should be 6 after increment
    
    // Test decrement
    c.decrement();
    assert(x == 5);  // x should be 5 after decrement
    
    c.decrement2();
    assert(x == 4);

    c.decrement3();
    assert(x == 3);
    return 0;
}
