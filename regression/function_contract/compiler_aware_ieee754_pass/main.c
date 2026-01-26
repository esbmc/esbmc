/* =============================================================================
 * TEST: IEEE 754 Fidelity in Contracts — Pillar 3 (Hardware Faithful)
 * =============================================================================
 *
 * PURPOSE:
 *   Verify that floating-point operations in ensures clauses use IEEE 754
 *   semantics (ieee_add2t) rather than mathematical addition. This ensures
 *   that contract verification is sound on real hardware where floating-point
 *   arithmetic is NOT associative/commutative.
 *
 * WHAT THIS TESTS:
 *   The normalize_fp_add_in_ensures pass converts add2t -> ieee_add2t in
 *   contract expressions, ensuring the solver uses hardware-faithful FP
 *   semantics for verification.
 *
 * SIGNIFICANCE FOR PAPER:
 *   This is a soundness differentiator. Tools that use mathematical reals
 *   for FP verification can produce unsound results on embedded systems
 *   where IEEE 754 rounding matters.
 *
 * EXPECTED: VERIFICATION SUCCESSFUL
 * =========================================================================== */

#include <assert.h>

/* Simple FP contract with addition */
double add_offset(double base, double offset) {
    __ESBMC_requires(offset >= 0.0);
    __ESBMC_requires(offset <= 100.0);
    __ESBMC_requires(base >= 0.0);
    __ESBMC_requires(base <= 1000.0);
    __ESBMC_ensures(__ESBMC_return_value == base + offset);

    return base + offset;
}

/* Contract with FP comparison */
double clamp_value(double val, double max_val) {
    __ESBMC_requires(max_val > 0.0);
    __ESBMC_ensures(__ESBMC_return_value <= max_val);
    __ESBMC_ensures(__ESBMC_return_value >= 0.0);

    if (val > max_val) return max_val;
    if (val < 0.0) return 0.0;
    return val;
}

int main() {
    double result1 = add_offset(1.0, 2.5);
    assert(result1 == 3.5);

    double result2 = clamp_value(150.0, 100.0);
    assert(result2 <= 100.0);
    assert(result2 >= 0.0);

    double result3 = clamp_value(-5.0, 100.0);
    assert(result3 == 0.0);

    return 0;
}
