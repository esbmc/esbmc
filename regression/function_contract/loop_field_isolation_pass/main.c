/* =============================================================================
 * TEST: Loop Field Isolation — Pillar 3 (Field-Specific Loop Havoc)
 * =============================================================================
 *
 * PURPOSE:
 *   Verify that when using loop invariants with structs, only the fields
 *   actually modified inside the loop body are havoc'd at the loop head.
 *   Fields not modified inside the loop should retain their pre-loop values.
 *
 * THE PROBLEM THIS SOLVES:
 *   Without field-level isolation, the loop invariant mechanism havocs the
 *   ENTIRE struct at the loop head. This destroys information about fields
 *   that are never written in the loop, causing the post-loop assertion
 *   to fail (false positive).
 *
 * EXPECTED: VERIFICATION SUCCESSFUL
 * =========================================================================== */

#include <assert.h>

typedef struct {
    int sum;
    int count;
    int config;  /* Never modified in the loop */
} Accumulator;

Accumulator acc;

int main() {
    acc.sum = 0;
    acc.count = 0;
    acc.config = 99;  /* Set once, never changed */

    __ESBMC_loop_invariant(
        acc.sum >= 0 && acc.sum <= 100 &&
        acc.count >= 0 && acc.count <= 10 &&
        acc.sum == acc.count * 10 &&
        acc.config == 99
    );
    while (acc.count < 10) {
        acc.sum += 10;
        acc.count++;
        /* acc.config is NOT modified */
    }

    assert(acc.sum == 100);
    assert(acc.count == 10);
    assert(acc.config == 99);  /* Must survive loop havoc */

    return 0;
}
