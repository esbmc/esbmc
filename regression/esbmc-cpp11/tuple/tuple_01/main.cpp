#include <cassert>
#include <tuple>

int main() {
    std::tuple<int, double, char> t1(42, 3.14, 'A');
    std::tuple<int, double, char> t2(100, 2.71, 'B');

    assert(std::get<0>(t1) == 42);
    assert(std::get<1>(t1) == 3.14);
    assert(std::get<2>(t1) == 'A');

    assert(std::get<0>(t2) == 100);
    assert(std::get<1>(t2) == 2.71);
    assert(std::get<2>(t2) == 'B');

    t2 = t1;

    assert(std::get<0>(t2) == 42);
    assert(std::get<1>(t2) == 3.14);
    assert(std::get<2>(t2) == 'A');

    t2 = std::tuple<int, double, char>(100, 2.71, 'B');
    t1.swap(t2);

    assert(std::get<0>(t1) == 100);
    assert(std::get<1>(t1) == 2.71);
    assert(std::get<2>(t1) == 'B');

    assert(std::get<0>(t2) == 42);
    assert(std::get<1>(t2) == 3.14);
    assert(std::get<2>(t2) == 'A');

    return 0;
}
