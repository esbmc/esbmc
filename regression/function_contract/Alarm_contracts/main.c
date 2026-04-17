// Alarm example converted to ESBMC function contract format
// Converted from experiment01/benchmarks_results/processed/Alarm
// Based on original Alarm.c with contracts added
// the design originally from https://github.com/lf-lang/lf-verifier-benchmarks. 
// I implemented the code in C.

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

// Time units in microseconds
#define SEC 1000000
#define TIME_BOUND (1 * SEC)

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

// Initialize controller
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

// Reaction 0: Startup/Computation
void reaction_0(Controller* self, long long time) {
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
        // Schedule turnOff action with CORRECT value
        self->turnOff_scheduled = true;
        self->turnOff_schedule_time = time + (1 * SEC);
        self->turnOff_value = 1; // FIXED: Now correctly set to 1
        
        self->out = 5;
        self->out2 = 10;
    }
}

// Reaction 1: Stop
bool reaction_1(Controller* self, long long time) {
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
    // Allocate Controller dynamically to use __ESBMC_is_fresh
    Controller* c = malloc(sizeof(Controller));
    if (c == NULL) {
        return 1;
    }
    
    init_controller(c);
    
    long long start_time = 0;
    current_time = start_time;
    
    // Execute reaction_0 at startup
    reaction_0(c, current_time);
    bool reaction_0_executed = true;
    
    // Property: G[0, 1 sec]((reaction_0) ==> F[0, 1 sec](reaction_1))
    // If reaction_0 executes, then reaction_1 must execute within 1 second
    
    if (reaction_0_executed) {
        // Advance time within the bound
        bool reaction_1_executed = false;
        
        // Check at scheduled time
        current_time = c->turnOff_schedule_time;
        
        // Verify time is within bound
        __ESBMC_assume(current_time <= start_time + TIME_BOUND);
        
        // Execute reaction_1
        reaction_1_executed = reaction_1(c, current_time);
        
        // Property assertion: reaction_1 should execute and successfully stop
        // Now with correct value (1), fault will be reset to 0
        // This assertion should PASS
        assert(reaction_1_executed && c->fault == 0 && 
               "machine_stops_within_1_sec: Machine should stop within 1 second");
    }
    
    free(c);
    return 0;
}

