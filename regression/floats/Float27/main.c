#include <stdio.h>
#include <assert.h>
#include <math.h>

// Manual implementation of ROUND_TO_AWAY for testing
double round_to_away(double value) {
    if (value >= 0.0) {
        return floor(value + 0.5);
    } else {
        return ceil(value - 0.5);
    }
}

// Manual implementation of round-to-even for comparison
double round_to_even(double value) {
    double rounded = floor(value + 0.5);
    // Check if we're exactly at a tie (x.5)
    if (fabs(value - floor(value) - 0.5) < 1e-10) {
        // We're at a tie, check if result is even
        long long int_part = (long long)rounded;
        if (int_part % 2 != 0) {
            // Result is odd, round toward even instead
            rounded = (value > 0) ? rounded - 1 : rounded + 1;
        }
    }
    return rounded;
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
    }
    
    return result;
}

void test_round_to_away_behavior() {
    printf("Testing ROUND_TO_AWAY behavior...\n");
    
    // Test positive ties
    assert(round_to_away(2.5) == 3.0);
    assert(round_to_away(1.5) == 2.0);
    assert(round_to_away(0.5) == 1.0);
    printf("✓ Positive ties round up (away from zero)\n");
    
    // Test negative ties
    assert(round_to_away(-2.5) == -3.0);
    assert(round_to_away(-1.5) == -2.0);
    assert(round_to_away(-0.5) == -1.0);
    printf("✓ Negative ties round down (away from zero)\n");
    
    // Test non-ties
    assert(round_to_away(2.4) == 2.0);
    assert(round_to_away(-2.4) == -2.0);
    assert(round_to_away(2.6) == 3.0);
    assert(round_to_away(-2.6) == -3.0);
    printf("✓ Non-ties work correctly\n");
}

void test_divide_and_round_away() {
    printf("\nTesting divide_and_round with ROUND_TO_AWAY...\n");
    
    // Test exact ties
    assert(divide_and_round_away(25, 10) == 3);  // 2.5 -> 3
    assert(divide_and_round_away(15, 10) == 2);  // 1.5 -> 2
    assert(divide_and_round_away(35, 10) == 4);  // 3.5 -> 4
    printf("✓ Exact ties round away from zero\n");
    
    // Test remainder > halfway
    assert(divide_and_round_away(26, 10) == 3);  // 2.6 -> 3
    assert(divide_and_round_away(17, 10) == 2);  // 1.7 -> 2
    printf("✓ Remainder > halfway rounds up\n");
    
    // Test remainder < halfway
    assert(divide_and_round_away(24, 10) == 2);  // 2.4 -> 2
    assert(divide_and_round_away(13, 10) == 1);  // 1.3 -> 1
    printf("✓ Remainder < halfway truncates\n");
    
    // Test exact division
    assert(divide_and_round_away(30, 10) == 3);
    assert(divide_and_round_away(20, 10) == 2);
    printf("✓ Exact division works correctly\n");
}

void compare_with_round_to_even() {
    printf("\nComparing ROUND_TO_AWAY vs ROUND_TO_EVEN...\n");
    
    printf("Value  | ROUND_TO_AWAY | ROUND_TO_EVEN | C round()\n");
    printf("-------|---------------|---------------|----------\n");
    
    double test_values[] = {1.5, 2.5, 3.5, 4.5, -1.5, -2.5, -3.5};
    int num_tests = sizeof(test_values) / sizeof(test_values[0]);
    
    for (int i = 0; i < num_tests; i++) {
        double val = test_values[i];
        double away_result = round_to_away(val);
        double even_result = round_to_even(val);
        double std_result = round(val);
        
        printf("%5.1f  |     %5.1f     |     %5.1f     |   %5.1f", 
               val, away_result, even_result, std_result);
               
        // Show when AWAY and EVEN differ
        if (away_result != even_result) {
            printf("  ← AWAY≠EVEN");
        }
        
        // Show that C round() matches AWAY
        if (away_result == std_result) {
            printf("  ← C=AWAY");
        }
        printf("\n");
    }
    
    printf("\nVerification:\n");
    // Verify that C round() actually uses round-away-from-zero
    assert(round(2.5) == 3.0);  // C round() rounds away from zero
    assert(round_to_away(2.5) == 3.0);  // Our implementation matches
    printf("✓ C round() and ROUND_TO_AWAY both give 3.0 for 2.5\n");
    
    // Show the difference with round-to-even
    assert(round_to_even(2.5) == 2.0);  // Round-to-even gives 2.0
    printf("✓ ROUND_TO_EVEN gives 2.0 for 2.5 (different from others)\n");
    
    printf("✓ C standard round() uses 'away from zero', same as ROUND_TO_AWAY\n");
}

int main() {
    test_round_to_away_behavior();
    test_divide_and_round_away();
    compare_with_round_to_even();
    
    printf("\nAll ROUND_TO_AWAY tests passed!\n");
    printf("Note: C round() uses 'away from zero' for ties, same as ROUND_TO_AWAY.\n");
    printf("The IEEE ROUND_TO_EVEN mode is different from C's round() function.\n");
    
    return 0;
}
