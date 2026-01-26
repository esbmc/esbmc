/* =============================================================================
 * TEST: Compiler-Aware Boolean Recovery — Pillar 1 (Semantic Restoration)
 * =============================================================================
 *
 * PURPOSE:
 *   Verify that ESBMC correctly handles ensures clauses containing complex
 *   boolean logic (&&, ||, !) that Clang decomposes into control flow with
 *   temporary variables. The inline_temporary_variables algorithm must
 *   reconstruct the original boolean expression from the GOTO IR.
 *
 * WHAT CLANG DOES:
 *   For `ensures(a && b || c)`, Clang generates:
 *     tmp$1 = a;
 *     if (!tmp$1) goto L1;
 *     tmp$2 = b;
 *     goto L2;
 *   L1: tmp$2 = 0;
 *   L2: tmp$3 = tmp$2;
 *     if (tmp$3) goto L3;
 *     tmp$4 = c;
 *     goto L4;
 *   L3: tmp$4 = 1;
 *   L4: assume(tmp$4);
 *
 *   ESBMC must inline these back to: assume(a && b || c)
 *
 * EXPECTED: VERIFICATION SUCCESSFUL
 * =========================================================================== */

#include <assert.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct {
    bool initialized;
    bool connected;
    int error_code;
} DeviceState;

/* Complex boolean ensures with short-circuit logic */
bool device_init(DeviceState *dev) {
    __ESBMC_requires(dev != NULL);
    __ESBMC_assigns(dev->initialized, dev->connected, dev->error_code);
    /* Complex boolean: either fully initialized OR has error */
    __ESBMC_ensures(
        (__ESBMC_return_value == true && dev->initialized == true && dev->connected == true) ||
        (__ESBMC_return_value == false && dev->error_code != 0)
    );

    /* Simulate successful init */
    dev->initialized = true;
    dev->connected = true;
    dev->error_code = 0;
    return true;
}

int main() {
    DeviceState dev;
    dev.initialized = false;
    dev.connected = false;
    dev.error_code = 0;

    bool result = device_init(&dev);

    /* Post-condition from contract */
    if (result) {
        assert(dev.initialized == true);
        assert(dev.connected == true);
    } else {
        assert(dev.error_code != 0);
    }

    return 0;
}
