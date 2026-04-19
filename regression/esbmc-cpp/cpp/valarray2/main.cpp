#include <cassert>
#include <valarray>

int nondet_int();

int main() {
    int x = nondet_int();
    __ESBMC_assume(x >= 0 && x <= 10);

    std::valarray<int> arr(x, 5); // valarray of size 5, filled with x
    assert(arr.size() == 5);

    // All elements should equal x
    for(size_t i = 0; i < arr.size(); ++i) {
        assert(arr[i] == x);
    }
}

