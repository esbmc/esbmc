#include <cassert>
#include <utility>

void test_pair() {
    std::pair<int, double> p1(10, 5.5);
    assert(p1.first == 10);
    assert(p1.second == 5.5);

    std::pair<int, double> p2 = std::make_pair(20, 3.14);
    assert(p2.first == 20);
    assert(p2.second == 3.14);

    std::pair<int, int> p3(1, 2);
    std::pair<int, int> p4(3, 4);
    p3.swap(p4);
    assert(p3.first == 3 && p3.second == 4);
    assert(p4.first == 1 && p4.second == 2);
}

void test_move() {
    int x = 42;
    int&& r = std::move(x);
    assert(r == 42);  // x is still valid, but moved-from
}

void test_swap() {
    int a = 5, b = 10;
    std::swap(a, b);
    assert(a == 10);
    assert(b == 5);
}

void test_forward() {
    auto func = [](auto&& x) -> decltype(auto) {
        return std::forward<decltype(x)>(x);
    };
    
    int val = 10;
    int&& ref = func(std::move(val));
    assert(ref == 10);
}

int main() {
    test_pair();
    test_move();
    test_swap();
    test_forward();
    return 0;
}

