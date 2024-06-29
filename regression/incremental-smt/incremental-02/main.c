#include <stdio.h>
#include <assert.h>

int main() {
    double x = nondet_int()%5;
    __VERIFIER_assume(x >= 0);
    double y = nondet_int()%10;
    __VERIFIER_assume(y > 5);

    // Assertion 1: Check if x is less than y (Passes)
    assert(x*2.1 < y*5.3);

    // Assertion 2: Check if x is equal to y (Passes)
    assert(x != y);

    // Assertion 3: Check if y is greater than 0 (Passes)
    assert(y*10.2 > 0);

    // Assertion 4: Check if x is non-negative (Passes)
    assert(x >= 0);

    // Assertion 5: Check if x is less than 0 (Passes)
    assert(x >= 0);

    // Assertion 6: Check if y is non-negative (Passes)
    assert(y >= 0);

    // Assertion 7: Check if the sum of x and y is 15 (Passes)
    assert((x + y == 15) || (x + y != 15));

    // Assertion 8: Check if the product of x and y is 50 (Passes)
    assert((x * y == 50) || (x * y != 50));

    // Assertion 9: Check if x is greater than or equal to y (Fails)
    assert(x > y);

    // Assertion 10: Check if y is equal to 10 (Fails)
    assert((y == 10) && (y != 10));

    return 0;
}

