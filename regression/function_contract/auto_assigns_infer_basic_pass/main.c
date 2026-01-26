/* =============================================================================
 * TEST: Auto Assigns Inference (Basic) — Pillar 2 (Automated Side-effect)
 * =============================================================================
 *
 * PURPOSE:
 *   When a function has requires/ensures but NO explicit __ESBMC_assigns,
 *   the system should automatically infer which memory locations the
 *   function body modifies. This test uses a simple function that modifies
 *   a global variable — the auto-inference should detect this and generate
 *   the correct assigns clause.
 *
 * CURRENT BEHAVIOR (before implementation):
 *   Without assigns, the contract system performs conservative havoc of ALL
 *   globals. This test verifies that auto-inference narrows the havoc to
 *   only the actually modified global.
 *
 * NOTE: This test currently uses explicit assigns. When auto-inference is
 *   implemented, the assigns clause should be removable while maintaining
 *   VERIFICATION SUCCESSFUL.
 *
 * EXPECTED: VERIFICATION SUCCESSFUL
 * =========================================================================== */

#include <assert.h>

int counter = 0;
int config_value = 42;

/* Explicit assigns for now; auto-inference target for future */
void increment_counter() {
    __ESBMC_assigns(counter);
    __ESBMC_ensures(counter == __ESBMC_old(counter) + 1);
    __ESBMC_ensures(config_value == __ESBMC_old(config_value));

    counter++;
}

int main() {
    counter = 10;
    config_value = 42;

    increment_counter();

    assert(counter == 11);
    assert(config_value == 42);  /* Must be preserved */

    return 0;
}
