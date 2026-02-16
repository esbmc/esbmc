#include <stdio.h>
#include <assert.h>
#include <math.h>

double round_to_away(double value) {
    if (value >= 0.0) {
        return floor(value + 0.5);
    } else {
        return ceil(value - 0.5);
    }
}

double round_to_even(double value) {
    double rounded = floor(value + 0.5);
    if (fabs(value - floor(value) - 0.5) < 1e-10) {
        long long int_part = (long long)rounded;
        if (int_part % 2 != 0) {
            rounded = (value > 0) ? rounded - 1 : rounded + 1;
        }
    }
    return rounded;
}

long long divide_and_round_away(long long fraction, long long factor) {
    long long remainder = fraction % factor;
    long long result = fraction / factor;
    
    if (remainder != 0) {
        long long factor_middle = factor / 2;
        
        if (remainder > factor_middle) {
            ++result;
        } else if (remainder == factor_middle) {
            ++result;
        }
    }
    
    return result;
}

void test_round_to_away_behavior() {
    assert(round_to_away(2.5) == 3.0);
    assert(round_to_away(1.5) == 2.0);
    assert(round_to_away(0.5) == 1.0);
    
    assert(round_to_away(-2.5) == -3.0);
    assert(round_to_away(-1.5) == -2.0);
    assert(round_to_away(-0.5) == -1.0);
    
    assert(round_to_away(2.4) == 2.0);
    assert(round_to_away(-2.4) == -2.0);
    assert(round_to_away(2.6) == 3.0);
    assert(round_to_away(-2.6) == -3.0);
}

void test_divide_and_round_away() {
    assert(divide_and_round_away(25, 10) == 3);  // 2.5 -> 3
    assert(divide_and_round_away(15, 10) == 2);  // 1.5 -> 2
    assert(divide_and_round_away(35, 10) == 4);  // 3.5 -> 4
    
    assert(divide_and_round_away(26, 10) == 3);  // 2.6 -> 3
    assert(divide_and_round_away(17, 10) == 2);  // 1.7 -> 2
    
    assert(divide_and_round_away(24, 10) == 2);  // 2.4 -> 2
    assert(divide_and_round_away(13, 10) == 1);  // 1.3 -> 1
    
    assert(divide_and_round_away(30, 10) == 3);
    assert(divide_and_round_away(20, 10) == 2);
}

void compare_with_round_to_even() {
    double test_values[] = {1.5, 2.5, 3.5, 4.5, -1.5, -2.5, -3.5};
    int num_tests = sizeof(test_values) / sizeof(test_values[0]);
    
    for (int i = 0; i < num_tests; i++) {
        double val = test_values[i];
        double away_result = round_to_away(val);
        double even_result = round_to_even(val);
        double std_result = round(val);
        
        printf("%5.1f  |     %5.1f     |     %5.1f     |   %5.1f", 
               val, away_result, even_result, std_result);
               
        if (away_result != even_result) {
            printf("  ← AWAY≠EVEN");
        }
        
        if (away_result == std_result) {
            printf("  ← C=AWAY");
        }
        printf("\n");
    }
    
    printf("\nVerification:\n");
    assert(round(2.5) == 3.0);  // C round() rounds away from zero
    assert(round_to_away(2.5) == 3.0);  // Our implementation matches
    
    assert(round_to_even(2.5) == 2.0);  // Round-to-even gives 2.0
}

int main() {
    test_round_to_away_behavior();
    test_divide_and_round_away();
    compare_with_round_to_even();
    return 0;
}
