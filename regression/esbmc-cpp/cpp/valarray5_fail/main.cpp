#include <cassert>
#include <valarray>

int nondet_int();

int main() {
    int a = nondet_int();
    int b = nondet_int();
    __ESBMC_assume(a >= 0 && a <= 10);
    __ESBMC_assume(b >= 0 && b <= 10);

    std::valarray<int> arr = {a, b};
    std::valarray<bool> mask = (arr % 2) == 0; // select even numbers

    std::valarray<int> evens = arr[mask];

    // Incorrect assumption: always at least one even
    assert(evens.size() >= 1); // <-- should FAIL
}

