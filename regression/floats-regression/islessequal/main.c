#include <math.h>

extern void __VERIFIER_error() __attribute__ ((__noreturn__));
void __VERIFIER_assert(int cond) { if (!(cond)) { ERROR: __VERIFIER_error(); } return; }
 
int main(void)
{
    __VERIFIER_assert(!islessequal(2.0, 1.0));
    __VERIFIER_assert(islessequal(1.0, 2.0));
    __VERIFIER_assert(islessequal(1.0, 1.0));
    __VERIFIER_assert(!islessequal(INFINITY, 1.0));
    __VERIFIER_assert(!islessequal(1.0, NAN));
 
    return 0;
}
