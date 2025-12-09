#include <cassert>
#include <memory>   // std::unique_ptr

struct MyClass {
    int value;
    MyClass(int val) : value(val) {}
    int getValue() { return value; }
};

int main() {
    std::unique_ptr<MyClass> p1(new MyClass(42));
    assert(p1.get() != nullptr);
    assert(p1->getValue() == 42);
    assert((*p1).getValue() == 42);

    std::unique_ptr<MyClass> p2(std::move(p1));
    assert(p1.get() == nullptr);
    assert(p2.get() != nullptr);
    assert(p2->getValue() == 42);

    return 0;
}
