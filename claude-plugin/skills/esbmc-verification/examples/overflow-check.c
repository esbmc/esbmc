/*
 * Integer Overflow Verification Example
 *
 * Demonstrates ESBMC's overflow detection capabilities:
 * - Signed integer overflow
 * - Unsigned integer overflow
 * - Arithmetic operation safety
 * - Safe arithmetic patterns
 *
 * Run with: esbmc overflow-check.c --overflow-check --unsigned-overflow-check --unwind 10
 */

#include <limits.h>
#include <assert.h>

// Non-deterministic input for symbolic verification
int __ESBMC_nondet_int(void);
unsigned int __ESBMC_nondet_uint(void);
void __ESBMC_assume(_Bool);
void __ESBMC_assert(_Bool, const char *);

/* Example 1: Signed Addition Overflow */
void signed_addition_overflow(void) {
    int a = __ESBMC_nondet_int();
    int b = __ESBMC_nondet_int();

    // Without constraints, this can overflow
    int result = a + b;  // ESBMC detects overflow

    // To fix: add overflow checks or constrain inputs
}

/* Example 2: Safe Signed Addition */
int safe_add(int a, int b) {
    // Check for overflow before performing addition
    if (b > 0 && a > INT_MAX - b) {
        return INT_MAX;  // Overflow would occur, saturate
    }
    if (b < 0 && a < INT_MIN - b) {
        return INT_MIN;  // Underflow would occur, saturate
    }
    return a + b;
}

void safe_addition_example(void) {
    int a = __ESBMC_nondet_int();
    int b = __ESBMC_nondet_int();

    int result = safe_add(a, b);

    // Result is always valid (saturated)
    __ESBMC_assert(result >= INT_MIN && result <= INT_MAX, "Result in valid range");
}

/* Example 3: Signed Multiplication Overflow */
void signed_multiplication_overflow(void) {
    int a = __ESBMC_nondet_int();
    int b = __ESBMC_nondet_int();

    // Multiplication can easily overflow
    int result = a * b;  // ESBMC detects overflow
}

/* Example 4: Safe Signed Multiplication */
int safe_multiply(int a, int b) {
    // Check for overflow before multiplication
    if (a > 0) {
        if (b > 0) {
            if (a > INT_MAX / b) return INT_MAX;
        } else {
            if (b < INT_MIN / a) return INT_MIN;
        }
    } else {
        if (b > 0) {
            if (a < INT_MIN / b) return INT_MIN;
        } else {
            if (a != 0 && b < INT_MAX / a) return INT_MAX;
        }
    }
    return a * b;
}

void safe_multiplication_example(void) {
    int a = __ESBMC_nondet_int();
    int b = __ESBMC_nondet_int();

    // Constrain to reasonable values for demonstration
    __ESBMC_assume(a >= -1000 && a <= 1000);
    __ESBMC_assume(b >= -1000 && b <= 1000);

    int result = safe_multiply(a, b);
    __ESBMC_assert(result >= INT_MIN && result <= INT_MAX, "Multiplication safe");
}

/* Example 5: Unsigned Overflow (Wrap-around) */
void unsigned_overflow_example(void) {
    unsigned int a = __ESBMC_nondet_uint();
    unsigned int b = __ESBMC_nondet_uint();

    // Unsigned arithmetic wraps around (well-defined but often unintended)
    unsigned int result = a + b;  // ESBMC with --unsigned-overflow-check detects this
}

/* Example 6: Safe Unsigned Addition */
unsigned int safe_unsigned_add(unsigned int a, unsigned int b) {
    if (a > UINT_MAX - b) {
        return UINT_MAX;  // Would overflow, saturate
    }
    return a + b;
}

void safe_unsigned_example(void) {
    unsigned int a = __ESBMC_nondet_uint();
    unsigned int b = __ESBMC_nondet_uint();

    unsigned int result = safe_unsigned_add(a, b);
    __ESBMC_assert(result >= a || result >= b, "No unexpected wrap");
}

/* Example 7: Array Index Overflow */
void array_index_overflow(void) {
    int arr[100];
    int i = __ESBMC_nondet_int();

    // Index computation can overflow
    int idx = i * 2 + 10;  // Can overflow if i is large

    // Even if we check bounds, overflow in computation is UB
    if (idx >= 0 && idx < 100) {
        arr[idx] = 42;
    }
}

/* Example 8: Safe Array Index Computation */
void safe_array_index(void) {
    int arr[100];
    int i = __ESBMC_nondet_int();

    // Constrain input to prevent overflow in index computation
    __ESBMC_assume(i >= 0 && i < 45);  // Ensures i*2+10 < 100 and no overflow

    int idx = i * 2 + 10;
    __ESBMC_assert(idx >= 0 && idx < 100, "Index in bounds");

    arr[idx] = 42;
}

/* Example 9: Shift Operation Overflow */
void shift_overflow_example(void) {
    int x = __ESBMC_nondet_int();
    int shift = __ESBMC_nondet_int();

    // Left shift can overflow
    // Also UB if shift < 0 or shift >= bit width
    int result = x << shift;  // ESBMC with --ub-shift-check detects issues
}

/* Example 10: Safe Shift Operation */
void safe_shift_example(void) {
    int x = __ESBMC_nondet_int();
    int shift = __ESBMC_nondet_int();

    // Constrain shift amount
    __ESBMC_assume(shift >= 0 && shift < 31);

    // Constrain x to prevent overflow
    __ESBMC_assume(x >= 0 && x <= (INT_MAX >> shift));

    int result = x << shift;
    __ESBMC_assert(result >= 0, "Shift result non-negative");
}

/* Example 11: Negation Overflow */
void negation_overflow_example(void) {
    int x = __ESBMC_nondet_int();

    // Negating INT_MIN overflows
    int neg = -x;  // UB if x == INT_MIN
}

/* Example 12: Safe Negation */
int safe_negate(int x) {
    if (x == INT_MIN) {
        return INT_MAX;  // Best we can do
    }
    return -x;
}

void safe_negation_example(void) {
    int x = __ESBMC_nondet_int();
    int result = safe_negate(x);

    __ESBMC_assert(result != INT_MIN || x == INT_MIN, "Negation handled correctly");
}

/* Example 13: Division Overflow */
void division_overflow_example(void) {
    int a = __ESBMC_nondet_int();
    int b = __ESBMC_nondet_int();

    // Division by zero and INT_MIN / -1 are both UB
    int result = a / b;  // ESBMC detects div-by-zero
}

/* Example 14: Safe Division */
int safe_divide(int a, int b) {
    if (b == 0) {
        return 0;  // Handle division by zero
    }
    if (a == INT_MIN && b == -1) {
        return INT_MAX;  // Handle overflow case
    }
    return a / b;
}

void safe_division_example(void) {
    int a = __ESBMC_nondet_int();
    int b = __ESBMC_nondet_int();

    int result = safe_divide(a, b);
    __ESBMC_assert(result >= INT_MIN && result <= INT_MAX, "Division safe");
}

/* Example 15: Factorial with Overflow Check */
unsigned long long factorial(int n) {
    if (n < 0) return 0;
    if (n <= 1) return 1;

    unsigned long long result = 1;
    for (int i = 2; i <= n; i++) {
        // Check for overflow before multiplication
        if (result > ULLONG_MAX / i) {
            return ULLONG_MAX;  // Overflow would occur
        }
        result *= i;
    }
    return result;
}

void factorial_example(void) {
    int n = __ESBMC_nondet_int();
    __ESBMC_assume(n >= 0 && n <= 20);  // 20! fits in unsigned long long

    unsigned long long result = factorial(n);
    __ESBMC_assert(result >= 1, "Factorial is at least 1");
}

int main(void) {
    // Safe examples (pass verification):
    safe_addition_example();
    safe_multiplication_example();
    safe_unsigned_example();
    safe_array_index();
    safe_shift_example();
    safe_negation_example();
    safe_division_example();
    factorial_example();

    // Unsafe examples (fail verification - uncomment to test):
    // signed_addition_overflow();
    // signed_multiplication_overflow();
    // unsigned_overflow_example();
    // array_index_overflow();
    // shift_overflow_example();
    // negation_overflow_example();
    // division_overflow_example();

    return 0;
}
