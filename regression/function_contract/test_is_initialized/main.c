// Test case for __ESBMC_is_initialized functionality
// This demonstrates the need for __ESBMC_is_initialized in non-initialization functions

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <assert.h>

// Time units in microseconds
#define SEC 1000000

// Controller state
typedef struct {
    int out;
    int out2;
    int fault;
    bool turnOff_scheduled;
    long long turnOff_schedule_time;
    int turnOff_value;
} Controller;

// Global time variable
long long current_time = 0;

// Initialization function: uses __ESBMC_is_fresh
void init_controller(Controller* self) {
    __ESBMC_requires(__ESBMC_is_fresh(&self, sizeof(Controller)));
    __ESBMC_requires(self != NULL);
    
    __ESBMC_ensures(self->out == 0 && self->out2 == 0 && self->fault == 0 && 
                    self->turnOff_scheduled == false && 
                    self->turnOff_schedule_time == 0 && self->turnOff_value == 0);
    
    self->out = 0;
    self->out2 = 0;
    self->fault = 0;
    self->turnOff_scheduled = false;
    self->turnOff_schedule_time = 0;
    self->turnOff_value = 0;
}

// Non-initialization function: uses __ESBMC_is_initialized
// This function expects an already-initialized Controller
void reaction_0(Controller* self, long long time) {
    __ESBMC_requires(__ESBMC_is_initialized(&self, sizeof(Controller)));
    __ESBMC_requires(self != NULL);
    __ESBMC_requires(time >= 0);
    
    __ESBMC_ensures(self->fault == 1);
    __ESBMC_ensures(self->turnOff_scheduled == true);
    __ESBMC_ensures(self->turnOff_schedule_time == time + 1000000);
    __ESBMC_ensures(self->turnOff_value == 1);
    __ESBMC_ensures(self->out == 5);
    __ESBMC_ensures(self->out2 == 10);
    
    // Operation
    self->fault = 1; // Fault occurs
    
    // Fault handling
    if (self->fault == 1) {
        // Schedule turnOff action
        self->turnOff_scheduled = true;
        self->turnOff_schedule_time = time + (1 * SEC);
        self->turnOff_value = 1;
        
        self->out = 5;
        self->out2 = 10;
    }
}

// Another non-initialization function
bool reaction_1(Controller* self, long long time) {
    __ESBMC_requires(__ESBMC_is_initialized(&self, sizeof(Controller)));
    __ESBMC_requires(self != NULL);
    __ESBMC_requires(time >= 0);
    __ESBMC_requires(self->turnOff_scheduled == true);
    __ESBMC_requires(self->turnOff_value == 1);
    __ESBMC_requires(time >= self->turnOff_schedule_time);
    
    __ESBMC_ensures(self->fault == 0);
    
    if (!self->turnOff_scheduled) {
        return false;
    }
    
    if (time >= self->turnOff_schedule_time) {
        // Trigger the alarm and reset fault
        if (self->turnOff_value == 1) {
            self->fault = 0;
            return true; // Successfully stopped
        }
        return false;
    }
    return false;
}

int main() {
    // Allocate Controller dynamically
    Controller* c = malloc(sizeof(Controller));
    if (c == NULL) {
        return 1;
    }
    
    init_controller(c);
    
    long long start_time = 0;
    current_time = start_time;
    
    // Execute reaction_0 at startup
    reaction_0(c, current_time);
    
    // Advance time to scheduled time
    current_time = c->turnOff_schedule_time;
    
    // Execute reaction_1
    bool reaction_1_executed = reaction_1(c, current_time);
    
    // Property assertion: reaction_1 should execute and successfully stop
    assert(reaction_1_executed && c->fault == 0 && 
           "Machine should stop within scheduled time");
    
    free(c);
    return 0;
}

