#include <assert.h>

unsigned nondet_uint();

int main()
{
    unsigned n = nondet_uint();  // n is nondeterministically assigned
    __ESBMC_assume(n > 0);  // Ensure that n is greater than 0, so the array size is valid

    char a[n];  // Variable-length array with size n
    unsigned k = sizeof(a);  // Total size of the array in bytes

    // Ensure that the total size of the array is n times the size of a char
    assert(k == n * sizeof(char));  // This is typically true, since sizeof(char) is 1 byte

    // Assertion to check the consistency of n and k (n == size of the array in elements)
    assert(n == k / sizeof(char));  // k / sizeof(char) should be equal to n (as sizeof(char) is usually 1)

    return 0;
}
