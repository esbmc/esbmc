/* Minimal test case for complex __ESBMC_old logic in replace-call mode causing verification failure
 * 
 * Bug: When --replace-call-with-contract is used with functions that have complex
 * ensures clauses containing multiple __ESBMC_old() calls in conditional expressions,
 * verification may fail even though the contract is correct.
 * 
 * This reproduces the issue from Ring.c::process_messages_at_time where
 * ensures clause uses multiple __ESBMC_old() in a conditional expression.
 */
#include <stdbool.h>

#define MAX_TIME 10

typedef struct {
    int value;
    int timestamp;
    bool valid;
} Message;

Message msg;
bool reaction_fired = false;
int reaction_fire_time = -1;

void process_reaction(int time) {
    __ESBMC_requires(time >= 0);
    __ESBMC_requires(time <= MAX_TIME);
    
    __ESBMC_assigns(reaction_fired, reaction_fire_time);
    __ESBMC_ensures(reaction_fired == true);
    __ESBMC_ensures(reaction_fire_time == time);
    
    reaction_fired = true;
    reaction_fire_time = time;
}

void process_at_time(int time) {
    __ESBMC_requires(time >= 0);
    __ESBMC_requires(time <= MAX_TIME);
    
    __ESBMC_assigns(msg.valid, msg.value, msg.timestamp, reaction_fired, reaction_fire_time);
    __ESBMC_ensures(msg.valid == false || msg.timestamp != time);
    // Complex ensures with multiple __ESBMC_old in conditional
    // Pattern from Ring.c::process_messages_at_time
    __ESBMC_ensures(!(__ESBMC_old(msg.valid) && __ESBMC_old(msg.timestamp) == time) || 
                    (reaction_fired == true && reaction_fire_time == time));
    
    // If message arrives at this time, process reaction
    if (msg.valid && msg.timestamp == time) {
        process_reaction(time);
        msg.valid = false;
    }
}

int main() {
    msg.value = 1;
    msg.timestamp = 5;
    msg.valid = true;
    
    process_at_time(5);
    
    // Property that should hold: if message was valid at time 5, reaction should fire
    assert(reaction_fired == true);
    assert(reaction_fire_time == 5);
    
    return 0;
}

