#include <math.h>
#include <assert.h>

/* Checking pow properties (approx.; due to floating point imprecision) */

int main()
{
    // Power laws: a^(b+c) = a^b * a^c
    double base = 2.5;
    double exp1_ = 3.0, exp2_ = 2.0;
    double left = pow(base, exp1_ + exp2_);
    double right = pow(base, exp1_) * pow(base, exp2_);
    assert(fabs(left - right) < 1e-10);
    
    // Power law: (a^b)^c = a^(b*c)
    double a = 2.0, b = 3.0, c = 2.0;
    double left2 = pow(pow(a, b), c);
    double right2 = pow(a, b * c);
    assert(fabs(left2 - right2) < 1e-10);
    
    // Negative exponent: a^(-n) = 1/(a^n)
    double base3 = 3.0, exp3 = 4.0;
    double left3 = pow(base3, -exp3);
    double right3 = 1.0 / pow(base3, exp3);
    assert(fabs(left3 - right3) < 1e-10);
}