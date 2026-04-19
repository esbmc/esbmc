#include <cassert>
#include <valarray>

int nondet_int();

int main() {
    int a = nondet_int(), b = nondet_int();
    __ESBMC_assume(a >= 0 && a <= 5);
    __ESBMC_assume(b >= 0 && b <= 5);

    std::valarray<int> arr = {a, b};
    int s = arr.sum();

    assert(s == a + b);   // Must hold
    assert(s >= 0);       // Since both are non-negative
}

