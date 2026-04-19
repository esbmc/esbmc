#include <stdio.h>
#include <assert.h>
#include <math.h>

int main() {
    double result1 = pow(4.0, 0.5);
    assert(result1 >= 1.999 && result1 <= 2.001);
    
    double result2 = pow(9.0, 0.5); 
    assert(result2 >= 2.999 && result2 <= 3.001);
    
    double result3 = pow(16.0, 0.5);
    assert(result3 >= 3.999 && result3 <= 4.001);
    
    double x1 = 0.01;
    double log_result1 = log1p(x1);  // ln(1 + 0.01)
    double expected1 = log(1.0 + x1);
    assert(fabs(log_result1 - expected1) < 1e-10);
    
    double x2 = -0.001;
    double log_result2 = log1p(x2);  // ln(1 - 0.001) 
    double expected2 = log(1.0 + x2);
    assert(fabs(log_result2 - expected2) < 1e-10);
    
    double result4 = pow(8.0, 1.0/3.0);
    assert(result4 >= 1.999 && result4 <= 2.001);
    
    double result5 = pow(27.0, 1.0/3.0);
    assert(result5 >= 2.999 && result5 <= 3.001);
    
    double a = 2.5;
    double b = 0.5; 
    double result6 = pow(a, b);  // Should be sqrt(2.5) ≈ 1.58
    assert(result6 >= 1.58 && result6 <= 1.59);
    
    double base = 4.0;
    double exponent = 0.5;
    double power_result = pow(base, exponent);
    assert(power_result >= 1.9 && power_result <= 2.1);
    
    double exp_result = exp(0.693147);  // Should be approximately 2.0
    assert(exp_result >= 1.999 && exp_result <= 2.001);
    
    return 0;
}
