#include <math.h>
#include <assert.h>

/* Edge cases */

int main()
{
    // Very small positive numbers (with proper tolerance for floating-point errors)
    double small_result = pow(1e-10, 2.0);
    assert(fabs(small_result - 1e-20) < 1e-35);  // Allow for representation errors
    
    // Numbers very close to 1
    double close_to_one = 1.0 + 1e-15;
    assert(pow(close_to_one, 2.0) > 1.0);
    
    // Test integer detection edge cases
    assert(pow(-2.0, 4.0) == 16.0);  // Even integer
    assert(pow(-2.0, 3.0) == -8.0);  // Odd integer
    
    // Test with exact binary representations to avoid floating-point issues
    assert(pow(0.125, 2.0) == 0.015625);  // 1/8 squared = 1/64 (exactly representable)
    assert(pow(0.25, 3.0) == 0.015625);   // 1/4 cubed = 1/64 (exactly representable)
}