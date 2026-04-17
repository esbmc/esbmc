#include <assert.h>
#include <stdbool.h>
#include <stddef.h>

#define MAX_TIME 10
#define INITIAL_PINGS 10

typedef struct {
    int pingsLeft;
    bool serve_scheduled_for_next_time;
} Ping_State;

typedef struct {
    int count;
} Pong_State;

Ping_State ping_state;
Pong_State pong_state;
bool ping_reaction_1_executed[MAX_TIME + 1];  // Track serve reaction execution
int current_time;

void init() {
    __ESBMC_assigns(ping_state.pingsLeft, ping_state.serve_scheduled_for_next_time, pong_state.count, current_time, ping_reaction_1_executed);
    __ESBMC_ensures(ping_state.pingsLeft == INITIAL_PINGS);
    __ESBMC_ensures(ping_state.serve_scheduled_for_next_time == false);
    __ESBMC_ensures(pong_state.count == 0);
    __ESBMC_ensures(current_time == 0);
    
    ping_state.pingsLeft = INITIAL_PINGS;
    ping_state.serve_scheduled_for_next_time = false;
    pong_state.count = 0;
    current_time = 0;
    
    for (int i = 0; i <= MAX_TIME; i++) {
        ping_reaction_1_executed[i] = false;
    }
}

// Execute all reactions at a given logical time
void execute_time_step(int time) {
    __ESBMC_requires(time >= 0);
    __ESBMC_requires(time <= MAX_TIME);
    
    __ESBMC_assigns(current_time, ping_reaction_1_executed, ping_state.pingsLeft, ping_state.serve_scheduled_for_next_time, pong_state.count);
    __ESBMC_ensures(current_time == time);
    
    current_time = time;
    int ping_output = -1;
    int pong_output = -1;
    bool ping_sends = false;
    bool pong_sends = false;
    
    // Step 1: Check if Ping's serve reaction should execute
    if (ping_state.serve_scheduled_for_next_time) {
        // Ping serve reaction executes
        ping_reaction_1_executed[time] = true;
        ping_output = ping_state.pingsLeft;
        ping_sends = true;
        ping_state.pingsLeft -= 1;
        ping_state.serve_scheduled_for_next_time = false;
    }
    
    // Step 2: If Ping sent, Pong receives (same logical time)
    if (ping_sends) {
        // Pong receive reaction
        pong_state.count += 1;
        pong_output = ping_output;
        pong_sends = true;
    }
    
    // Step 3: If Pong sent, Ping receives (same logical time)
    if (pong_sends) {
        // Ping receive reaction
        if (ping_state.pingsLeft > 0) {
            // Schedule serve for time + 1 nsec (due to logical action delay)
            ping_state.serve_scheduled_for_next_time = true;
        }
    }
}

// Property: G[0, 4 nsec](ping_reaction_1 ==> X(!ping_reaction_1))
// "Globally in [0,4 nsec], if ping_reaction_1 executes at time t,
// then it does NOT execute at time t+1"
// expect=false means we EXPECT this to be violated
void check_property() {
    __ESBMC_assigns();  // Pure function, no side effects
    // Add ensures to help verification
    __ESBMC_ensures(true);  // Always true, helps ESBMC understand this is a pure check
    
    bool property_violated = false;
    
    for (int t = 0; t <= 4 && t < MAX_TIME; t++) {
        if (ping_reaction_1_executed[t] && ping_reaction_1_executed[t + 1]) {
            // Found two consecutive executions in [0,4] range
            property_violated = true;
            break;
        }
    }
    
    // Since expect=false, we EXPECT the property to be violated
    // So if property is violated, that's the expected behavior
    // For ESBMC verification: assert that violation occurs
    assert(property_violated); // Should be true (property IS violated)
}

int main() {
    init();
    
    // Time 0: Startup reaction schedules serve for time 0 + 1 nsec = time 1
    ping_state.serve_scheduled_for_next_time = true;
    
    // Execute time steps
    for (int t = 1; t <= MAX_TIME; t++) {
        execute_time_step(t);
    }
    
    check_property();
    
    return 0;
}