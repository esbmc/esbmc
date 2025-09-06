#include <stdio.h>
#include <assert.h>
#include <math.h>

// Manual implementation of ROUND_TO_AWAY for testing
double round_to_away(double value) {
    if (value >= 0.0) {
        // For positive numbers: round ties away from zero (upward)
        return floor(value + 0.5);
    } else {
        // For negative numbers: round ties away from zero (downward) 
        return ceil(value - 0.5);
    }
}

// Test the divide_and_round logic with ROUND_TO_AWAY
long long divide_and_round_away(long long fraction, long long factor) {
    long long remainder = fraction % factor;
    long long result = fraction / factor;
    
    if (remainder != 0) {
        long long factor_middle = factor / 2;
        
        if (remainder > factor_middle) {
            ++result;
        } else if (remainder == factor_middle) {
            ++result;  // always round away from zero for ties
        }
        // remainder < factor_middle: crop (do nothing)
    }
    
    return result;
}

void test_round_to_away_behavior() {
    printf("Testing ROUND_TO_AWAY behavior...\n");
    
    // Test 1: Positive ties (should round away from zero = up)
    assert(round_to_away(2.5) == 3.0);
    assert(round_to_away(1.5) == 2.0);
    assert(round_to_away(0.5) == 1.0);
    printf("✓ Positive ties round up (away from zero)\n");
    
    // Test 2: Negative ties (should round away from zero = down)
    assert(round_to_away(-2.5) == -3.0);
    assert(round_to_away(-1.5) == -2.0);
    assert(round_to_away(-0.5) == -1.0);
    printf("✓ Negative ties round down (away from zero)\n");
    
    // Test 3: Non-ties less than halfway (should truncate)
    assert(round_to_away(2.4) == 2.0);
    assert(round_to_away(-2.4) == -2.0);
    printf("✓ Values less than halfway truncate\n");
    
    // Test 4: Non-ties more than halfway (should round toward infinity)
    assert(round_to_away(2.6) == 3.0);
    assert(round_to_away(-2.6) == -3.0);
    printf("✓ Values more than halfway round toward infinity\n");
}

void test_divide_and_round_away() {
    printf("\nTesting divide_and_round with ROUND_TO_AWAY...\n");
    
    // Test exact ties (remainder == factor/2)
    assert(divide_and_round_away(25, 10) == 3);  // 25/10 = 2.5 -> 3
    assert(divide_and_round_away(15, 10) == 2);  // 15/10 = 1.5 -> 2
    assert(divide_and_round_away(35, 10) == 4);  // 35/10 = 3.5 -> 4
    printf("✓ Exact ties round away from zero\n");
    
    // Test remainder > factor/2 (should round up)
    assert(divide_and_round_away(26, 10) == 3);  // 26/10 = 2.6 -> 3
    assert(divide_and_round_away(17, 10) == 2);  // 17/10 = 1.7 -> 2
    printf("✓ Remainder > halfway rounds up\n");
    
    // Test remainder < factor/2 (should truncate)
    assert(divide_and_round_away(24, 10) == 2);  // 24/10 = 2.4 -> 2
    assert(divide_and_round_away(13, 10) == 1);  // 13/10 = 1.3 -> 1
    printf("✓ Remainder < halfway truncates\n");
    
    // Test exact division (no remainder)
    assert(divide_and_round_away(30, 10) == 3);  // 30/10 = 3.0 -> 3
    assert(divide_and_round_away(20, 10) == 2);  // 20/10 = 2.0 -> 2
    printf("✓ Exact division works correctly\n");
}

void compare_with_round_to_even() {
    printf("\nComparing ROUND_TO_AWAY vs standard rounding...\n");
    
    // Standard C round() uses round-to-nearest-ties-to-even
    printf("Value  | ROUND_TO_AWAY | Standard round()\n");
    printf("-------|---------------|------------------\n");
    
    double test_values[] = {1.5, 2.5, 3.5, 4.5, -1.5, -2.5, -3.5};
    int num_tests = sizeof(test_values) / sizeof(test_values[0]);
    
    for (int i = 0; i < num_tests; i++) {
        double val = test_values[i];
        double away_result = round_to_away(val);
        double std_result = round(val);
        
        printf("%5.1f  |     %5.1f     |      %5.1f", 
               val, away_result, std_result);
               
        // Show when they differ
        if (away_result != std_result) {
            printf("  ← DIFFERENT");
        }
        printf("\n");
    }
    
    // Verify specific differences
    printf("\nKey differences:\n");
    assert(round_to_away(2.5) == 3.0 && round(2.5) == 2.0);
    printf("✓ 2.5: AWAY=3.0, EVEN=2.0\n");
    
    assert(round_to_away(-2.5) == -3.0 && round(-2.5) == -2.0);
    printf("✓ -2.5: AWAY=-3.0, EVEN=-2.0\n");
    
    // Cases where they happen to agree
    assert(round_to_away(1.5) == 2.0 && round(1.5) == 2.0);
    printf("✓ 1.5: Both give 2.0 (coincidentally same)\n");
}

int main() {
    test_round_to_away_behavior();
    test_divide_and_round_away();
    compare_with_round_to_even();
    
    printf("\nAll ROUND_TO_AWAY tests passed!\n");
    printf("The implementation correctly rounds ties away from zero.\n");
    
    return 0;
}

