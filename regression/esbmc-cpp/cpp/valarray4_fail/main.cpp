#include <cassert>
#include <valarray>

int nondet_int();

int main() {
    int a = nondet_int();
    int b = nondet_int();
    int c = nondet_int();
    __ESBMC_assume(a >= 0 && a <= 3);
    __ESBMC_assume(b >= 0 && b <= 3);
    __ESBMC_assume(c >= 0 && c <= 3);

    std::valarray<int> arr = {a, b, c};
    std::slice sl(1, 2, 1); // Take elements at index 1 and 2
    std::valarray<int> sub = arr[sl];

    assert(sub.size() == 1);
    assert(sub[0] == b);
    assert(sub[1] == c);
}

