/* init_controller_fail: Test __ESBMC_is_fresh with incorrect implementation
 * Tests that __ESBMC_is_fresh allocates fresh memory but function violates ensures clause
 * This test should be run with: --function init_controller --enforce-contract init_controller
 */
#include <stdbool.h>
#include <stdlib.h>

typedef struct {
    int out;
    int out2;
    int fault;
    bool turnOff_scheduled;
    int turnOff_schedule_time;
    int turnOff_value;
} Controller;

void init_controller(Controller *self) {
    __ESBMC_requires(self != NULL);
    __ESBMC_requires(__ESBMC_is_fresh(&self, sizeof(Controller)));
    
    __ESBMC_ensures(self->out == 0 && self->out2 == 0 && self->fault == 0 && 
                    self->turnOff_scheduled == false && 
                    self->turnOff_schedule_time == 0 && self->turnOff_value == 0);
    
    // WRONG: sets out = 1, but ensures requires out == 0
    self->out = 1;
    self->out2 = 0;
    self->fault = 0;
    self->turnOff_scheduled = false;
    self->turnOff_schedule_time = 0;
    self->turnOff_value = 0;
}

