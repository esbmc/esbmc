/* =============================================================================
 * TEST: Multi-Module Modular Verification — Pillar 4 (Unified Framework)
 * =============================================================================
 *
 * PURPOSE:
 *   Demonstrate complete modular verification workflow: verify each function
 *   against its contract (enforce), then replace calls with contracts (replace)
 *   to verify the system as a whole. This combines:
 *   - Function contracts (horizontal decomposition)
 *   - Field-sensitive assigns for minimal havoc
 *   - __ESBMC_old for state transition verification
 *   - __ESBMC_return_value for return type verification
 *
 * ARCHITECTURE:
 *   HAL layer -> Driver layer -> Application layer
 *   Each layer is verified against its contract independently.
 *
 * EXPECTED: VERIFICATION SUCCESSFUL
 * =========================================================================== */

#include <assert.h>
#include <stddef.h>
#include <stdbool.h>

/* ============== HAL Layer ============== */

typedef struct {
    int reg_status;
    int reg_data;
    int reg_control;
} HW_Registers;

HW_Registers hw;

/* HAL: Read data register if status is ready */
int hal_read_data() {
    __ESBMC_requires(hw.reg_status == 1);
    __ESBMC_assigns(hw.reg_status);
    __ESBMC_ensures(hw.reg_status == 0);
    __ESBMC_ensures(__ESBMC_return_value == __ESBMC_old(hw.reg_data));

    int data = hw.reg_data;
    hw.reg_status = 0;  /* Mark as consumed */
    return data;
}

/* ============== Driver Layer ============== */

typedef struct {
    int last_value;
    int read_count;
    bool has_data;
} DriverState;

DriverState drv;

/* Driver: Poll and read from HAL */
void driver_poll() {
    __ESBMC_requires(hw.reg_status == 1);
    __ESBMC_assigns(hw.reg_status, drv.last_value, drv.read_count, drv.has_data);
    __ESBMC_ensures(drv.has_data == true);
    __ESBMC_ensures(drv.read_count == __ESBMC_old(drv.read_count) + 1);

    drv.last_value = hal_read_data();
    drv.read_count++;
    drv.has_data = true;
}

/* ============== Application Layer ============== */

int app_result;

/* Application: Process driver data */
void app_process() {
    __ESBMC_requires(drv.has_data == true);
    __ESBMC_assigns(app_result, drv.has_data);
    __ESBMC_ensures(app_result == drv.last_value * 2);
    __ESBMC_ensures(drv.has_data == false);

    app_result = drv.last_value * 2;
    drv.has_data = false;
}

/* ============== System Integration ============== */

int main() {
    /* Setup: hardware has data ready */
    hw.reg_status = 1;
    hw.reg_data = 42;
    hw.reg_control = 0;

    drv.last_value = 0;
    drv.read_count = 0;
    drv.has_data = false;

    app_result = 0;

    /* Execute the pipeline */
    driver_poll();

    assert(drv.has_data == true);
    assert(drv.read_count == 1);

    app_process();

    assert(app_result == 84);  /* 42 * 2 */
    assert(drv.has_data == false);

    return 0;
}
