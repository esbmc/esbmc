#include <stdio.h>
#include <math.h>
#include <assert.h>

extern void __VERIFIER_error() __attribute__ ((__noreturn__)) { assert(0); };
void __VERIFIER_assert(int cond) { if (!(cond)) { ERROR: __VERIFIER_error(); } return; }
 
int main(void)
{
    __VERIFIER_assert(islessgreater(2.0, 1.0));
    __VERIFIER_assert(islessgreater(1.0, 2.0));
    __VERIFIER_assert(!islessgreater(1.0, 1.0));
    __VERIFIER_assert(islessgreater(INFINITY, 1.0));
    __VERIFIER_assert(!islessgreater(1.0, NAN));
 
    return 0;
}
