/* =============================================================================
 * TEST: Auto Assigns with Conditional Modification — Pillar 2
 * =============================================================================
 *
 * PURPOSE:
 *   Test that assigns inference handles conditional writes correctly.
 *   A function that CONDITIONALLY modifies a field must still list that
 *   field in assigns (since it MAY be modified on some paths).
 *
 * TECHNICAL INSIGHT:
 *   Sound auto-inference must be path-insensitive at the assigns level:
 *   if ANY path writes to a location, it must be in the assigns set.
 *   But the ensures clause can be path-sensitive, constraining the value
 *   based on conditions.
 *
 * EXPECTED: VERIFICATION SUCCESSFUL
 * =========================================================================== */

#include <assert.h>
#include <stddef.h>

typedef struct {
    int value;
    int error_code;
    int access_count;
} Register;

/* Conditionally modifies error_code; always increments access_count */
void safe_write(Register *reg, int new_val) {
    __ESBMC_requires(reg != NULL);
    __ESBMC_requires(new_val >= 0);
    __ESBMC_assigns(reg->value, reg->error_code, reg->access_count);
    __ESBMC_ensures(reg->access_count == __ESBMC_old(reg->access_count) + 1);
    __ESBMC_ensures(new_val <= 100 ? reg->value == new_val : 
                    reg->value == __ESBMC_old(reg->value));
    __ESBMC_ensures(new_val > 100 ? reg->error_code == 1 : 
                    reg->error_code == __ESBMC_old(reg->error_code));

    reg->access_count++;

    if (new_val <= 100) {
        reg->value = new_val;
    } else {
        reg->error_code = 1;  /* Range error */
    }
}

int main() {
    Register r;
    r.value = 50;
    r.error_code = 0;
    r.access_count = 0;

    safe_write(&r, 75);

    assert(r.value == 75);
    assert(r.error_code == 0);
    assert(r.access_count == 1);

    return 0;
}
