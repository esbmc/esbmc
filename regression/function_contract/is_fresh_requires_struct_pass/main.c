/* is_fresh_requires_struct_pass: structural twin of init_controller_pass that
 * uses the canonical bare-pointer form is_fresh(self, sizeof(*self)) instead
 * of the address-of-variable form is_fresh(&self, sizeof(Controller)).
 *
 * Both forms must produce equivalent allocations.  This guards against future
 * regressions in the requires-side codegen for the documented form.
 */
#include <stdbool.h>
#include <stdlib.h>

typedef struct {
    int  out;
    int  out2;
    int  fault;
    bool turnOff_scheduled;
    int  turnOff_schedule_time;
    int  turnOff_value;
} Controller;

__ESBMC_contract
void init_controller(Controller *self)
{
    __ESBMC_requires(__ESBMC_is_fresh(self, sizeof(Controller)));

    __ESBMC_ensures(self->out == 0 && self->out2 == 0 && self->fault == 0 &&
                    self->turnOff_scheduled == false &&
                    self->turnOff_schedule_time == 0 &&
                    self->turnOff_value == 0);

    self->out = 0;
    self->out2 = 0;
    self->fault = 0;
    self->turnOff_scheduled = false;
    self->turnOff_schedule_time = 0;
    self->turnOff_value = 0;
}

int main(void) { return 0; }
