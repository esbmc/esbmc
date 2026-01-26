/* =============================================================================
 * TEST: Array of Structs — Pillar 1 (Access Path Restoration)
 * =============================================================================
 *
 * PURPOSE:
 *   Test that assigns can target a specific field of a specific element in
 *   an array of structs: arr[i].field. This combines index-based access
 *   with member-level access path tracking.
 *
 * TECHNICAL CHALLENGE:
 *   The access path is: base_array + (index * sizeof(Element)) + field_offset.
 *   The restorer must decompose this compound offset into:
 *     index2tc(arr, i) -> member2tc(.field)
 *
 * REAL-WORLD RELEVANCE:
 *   Sensor arrays in automotive/aerospace: each sensor has multiple fields
 *   (reading, calibration, status), and a function may only update the reading
 *   of one sensor without affecting others.
 *
 * EXPECTED: VERIFICATION SUCCESSFUL
 * =========================================================================== */

#include <assert.h>
#include <stddef.h>

#define NUM_SENSORS 3

typedef struct {
    int reading;
    int calibration;
    int status;
} Sensor;

Sensor sensors[NUM_SENSORS];

/* Updates reading of a single sensor, preserving all other state */
void update_reading(int idx, int new_reading) {
    __ESBMC_requires(idx >= 0 && idx < NUM_SENSORS);
    __ESBMC_assigns(sensors[idx].reading);
    __ESBMC_ensures(sensors[idx].reading == new_reading);

    sensors[idx].reading = new_reading;
}

int main() {
    /* Initialize all sensors */
    sensors[0].reading = 10;  sensors[0].calibration = 100;  sensors[0].status = 1;
    sensors[1].reading = 20;  sensors[1].calibration = 200;  sensors[1].status = 1;
    sensors[2].reading = 30;  sensors[2].calibration = 300;  sensors[2].status = 1;

    /* Update only sensor[1].reading */
    update_reading(1, 999);

    /* The updated sensor's reading changed */
    assert(sensors[1].reading == 999);

    /* Other sensors are completely untouched */
    assert(sensors[0].reading == 10);
    assert(sensors[0].calibration == 100);
    assert(sensors[2].reading == 30);
    assert(sensors[2].calibration == 300);

    /* Even the updated sensor's other fields are preserved */
    assert(sensors[1].calibration == 200);
    assert(sensors[1].status == 1);

    return 0;
}
