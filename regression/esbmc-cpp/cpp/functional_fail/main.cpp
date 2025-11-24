#include <functional>
#include <cassert>

int add(int x, int y) { return x + y; }

int main() {
    std::function<int(int, int)> func = add;  
    std::function<int(int, int)> func2 = [](int x, int y) { return x * y; };

    assert(func(2, 3) != 5);
    assert(func2(3, 4) == 12);

    return 0;
}
