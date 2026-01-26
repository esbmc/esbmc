/* =============================================================================
 * TEST: Nested Loop Field Isolation — Pillar 3 (Field-Specific Loop Havoc)
 * =============================================================================
 *
 * PURPOSE:
 *   Test field-level loop havoc with nested struct and a loop that only
 *   modifies one sub-field. The invariant should preserve unmodified fields
 *   across the loop boundary.
 *
 * SIGNIFICANCE:
 *   This models a common embedded pattern: a state machine struct where
 *   a timer loop updates one counter while the configuration fields remain
 *   static. False positives here would block verification of real firmware.
 *
 * EXPECTED: VERIFICATION SUCCESSFUL
 * =========================================================================== */

#include <assert.h>

typedef struct {
    int timer_count;
    int timer_limit;
} Timer;

typedef struct {
    Timer hw_timer;
    int device_id;      /* Never modified */
    int firmware_ver;   /* Never modified */
} Device;

Device dev;

int main() {
    dev.hw_timer.timer_count = 0;
    dev.hw_timer.timer_limit = 5;
    dev.device_id = 0x1234;
    dev.firmware_ver = 3;

    __ESBMC_loop_invariant(
        dev.hw_timer.timer_count >= 0 &&
        dev.hw_timer.timer_count <= dev.hw_timer.timer_limit &&
        dev.hw_timer.timer_limit == 5 &&
        dev.device_id == 0x1234 &&
        dev.firmware_ver == 3
    );
    while (dev.hw_timer.timer_count < dev.hw_timer.timer_limit) {
        dev.hw_timer.timer_count++;
    }

    assert(dev.hw_timer.timer_count == 5);
    assert(dev.device_id == 0x1234);     /* Preserved across loop */
    assert(dev.firmware_ver == 3);       /* Preserved across loop */

    return 0;
}
