#include <math.h>

extern void __VERIFIER_error() __attribute__ ((__noreturn__));
void __VERIFIER_assert(int cond) { if (!(cond)) { ERROR: __VERIFIER_error(); } return; }
 
int main(void)
{
    __VERIFIER_assert(isunordered(NAN, 1.0));
    __VERIFIER_assert(isunordered(1.0, NAN));
    __VERIFIER_assert(isunordered(NAN, NAN));
    __VERIFIER_assert(!isunordered(1.0, 0.0));
 
    return 0;
}
