#include <assert.h>
#include <float.h>
#include <math.h>
 
int main() {
    double x = nondet_double();
    double y = nondet_double();
 
    __ESBMC_assume(x > 0.0 && x < 1.0e-154);
    __ESBMC_assume(y > 0.0 && y < 1.0e-154);
 
    double product = x * y;
    double ulp;
 
    if (product < DBL_MIN) {
        // Force underflow
        product = nondet_double();
        __ESBMC_assume(product == 0.0);
        ulp = DBL_MIN * 0x1p-52;
    } else {
        ulp = product * 0x1p-52;
    }
 
    // Model IEEE-754 rounding error
    double rounding_error = nondet_double();
    __ESBMC_assume(rounding_error >= -ulp/2 && rounding_error <= ulp/2);
 
    double result = product + rounding_error;
 
    // Underflow check
    assert(result > 0.0);
 
    return 0;
}

