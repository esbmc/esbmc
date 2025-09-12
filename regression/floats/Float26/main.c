#include <stdio.h>
#include <assert.h>
#include <math.h>

int main() {
    double result1 = pow(4.0, 0.5);
    assert(result1 > 2 && result1 < 2.001);
    
    double result3 = pow(16.0, 0.5);
    assert(result3 > 4 && result3 < 4.001);
    
    double x1 = 0.01;
    double log_result1 = log1p(x1);
    double expected1 = log(1.0 + x1);
    assert(fabs(log_result1 - expected1) < 1e-20);
    
    double x2 = -0.001;
    double log_result2 = log1p(x2);
    double expected2 = log(1.0 + x2);
    assert(fabs(log_result2 - expected2) < 1e-20);
    
    double exp_result = exp(0.693147);
    assert(exp_result > 2 && exp_result < 2.001);
    
    return 0;
}
