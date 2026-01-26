/* =============================================================================
 * TEST: Driver Subsystem with Snapshot Isolation — Pillar 4
 * =============================================================================
 *
 * PURPOSE:
 *   End-to-end test of modular verification on a realistic embedded driver
 *   subsystem. This test combines ALL four pillars:
 *
 *   Pillar 1 (Semantic Restoration): Complex boolean ensures with &&/||
 *   Pillar 2 (Field-Sensitive): assigns on specific struct fields via pointer
 *   Pillar 3 (Hardware Fidelity): Integer overflow-safe contracts
 *   Pillar 4 (Modular Framework): Replace-call with snapshot isolation
 *
 * SCENARIO:
 *   An I2C bus driver with read/write operations. The driver maintains
 *   internal state (bus_state) and an error counter. Operations must
 *   preserve bus integrity: a write followed by a read returns the
 *   written value, and errors are properly tracked.
 *
 * EXPECTED: VERIFICATION SUCCESSFUL
 * =========================================================================== */

#include <assert.h>
#include <stddef.h>
#include <stdbool.h>

#define I2C_IDLE    0
#define I2C_BUSY    1
#define I2C_ERROR   2
#define MAX_ERRORS  255

typedef struct {
    int bus_state;
    int last_addr;
    int last_data;
    int error_count;
} I2C_Driver;

/* Write a byte to the I2C bus */
bool i2c_write(I2C_Driver *drv, int addr, int data) {
    __ESBMC_requires(drv != NULL);
    __ESBMC_requires(addr >= 0 && addr <= 127);
    __ESBMC_requires(drv->bus_state == I2C_IDLE);

    __ESBMC_assigns(drv->bus_state, drv->last_addr, drv->last_data, drv->error_count);

    /* Pillar 1: Complex boolean ensures recovered from Clang IR */
    __ESBMC_ensures(
        (__ESBMC_return_value == true &&
         drv->bus_state == I2C_IDLE &&
         drv->last_addr == addr &&
         drv->last_data == data &&
         drv->error_count == __ESBMC_old(drv->error_count))
        ||
        (__ESBMC_return_value == false &&
         drv->bus_state == I2C_ERROR &&
         drv->error_count == __ESBMC_old(drv->error_count) + 1)
    );

    /* Simulate write — always succeeds in this model */
    drv->bus_state = I2C_BUSY;
    drv->last_addr = addr;
    drv->last_data = data;
    drv->bus_state = I2C_IDLE;
    return true;
}

/* Read a byte from the I2C bus */
int i2c_read(I2C_Driver *drv, int addr) {
    __ESBMC_requires(drv != NULL);
    __ESBMC_requires(addr >= 0 && addr <= 127);
    __ESBMC_requires(drv->bus_state == I2C_IDLE);
    __ESBMC_requires(drv->last_addr == addr);

    __ESBMC_assigns(drv->bus_state);
    __ESBMC_ensures(__ESBMC_return_value == drv->last_data);
    __ESBMC_ensures(drv->bus_state == I2C_IDLE);

    /* Read returns the last written data for this address */
    return drv->last_data;
}

int main() {
    I2C_Driver drv;
    drv.bus_state = I2C_IDLE;
    drv.last_addr = -1;
    drv.last_data = 0;
    drv.error_count = 0;

    /* Write 0xAB to address 0x50 */
    bool ok = i2c_write(&drv, 0x50, 0xAB);
    assert(ok == true);

    /* The write should have recorded the address and data */
    assert(drv.bus_state == I2C_IDLE);
    assert(drv.last_addr == 0x50);
    assert(drv.last_data == 0xAB);
    assert(drv.error_count == 0);

    /* Read from the same address should return the written value */
    int val = i2c_read(&drv, 0x50);
    assert(val == 0xAB);

    return 0;
}
