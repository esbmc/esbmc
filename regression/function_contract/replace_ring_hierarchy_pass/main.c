#include <assert.h>
#include <stdbool.h>
#include <stddef.h>

#define MAX_TIME 10

typedef struct {
    int value;
    int timestamp;
    bool valid;
} Message;

typedef struct {
    int received;
    bool reaction_2_fired;
    int reaction_2_fire_time;
} Source;

typedef struct {
    int value;
} Node;

// Global time
int current_time = 0;

// Ring components
Source source;
Node n1, n2, n3, n4, n5;

// Message buffers (with 1 nsec delay)
Message msg_s_to_n1 = {0, -1, false};
Message msg_n1_to_n2 = {0, -1, false};
Message msg_n2_to_n3 = {0, -1, false};
Message msg_n3_to_n4 = {0, -1, false};
Message msg_n4_to_n5 = {0, -1, false};
Message msg_n5_to_s = {0, -1, false};

void source_startup() {
    __ESBMC_assigns(source.received, source.reaction_2_fired, source.reaction_2_fire_time);
    __ESBMC_ensures(source.received == 0);
    __ESBMC_ensures(source.reaction_2_fired == false);
    __ESBMC_ensures(source.reaction_2_fire_time == -1);
    
    source.received = 0;
    source.reaction_2_fired = false;
    source.reaction_2_fire_time = -1;
}

void source_reaction_start() {
    __ESBMC_requires(current_time >= 0);
    
    __ESBMC_assigns(msg_s_to_n1.value, msg_s_to_n1.timestamp, msg_s_to_n1.valid);
    __ESBMC_ensures(msg_s_to_n1.value == __ESBMC_old(source.received));
    __ESBMC_ensures(msg_s_to_n1.timestamp == __ESBMC_old(current_time) + 1);
    __ESBMC_ensures(msg_s_to_n1.valid == true);
    
    // Output the received value
    msg_s_to_n1.value = source.received;
    msg_s_to_n1.timestamp = current_time + 1; // 1 nsec delay
    msg_s_to_n1.valid = true;
}

void source_reaction_in(int in_value, int time) {
    __ESBMC_requires(time >= 0);
    __ESBMC_requires(time <= MAX_TIME);
    
    __ESBMC_assigns(source.received, source.reaction_2_fired, source.reaction_2_fire_time);
    __ESBMC_ensures(source.received == in_value);
    __ESBMC_ensures(source.reaction_2_fired == true);
    // Use parameter instead of __ESBMC_old to avoid global variable issues
    __ESBMC_ensures(source.reaction_2_fire_time == time);
    __ESBMC_ensures(source.reaction_2_fire_time >= 0 && source.reaction_2_fire_time <= MAX_TIME);
    
    // This is Ring_s_reaction_2
    source.received = in_value;
    source.reaction_2_fired = true;
    source.reaction_2_fire_time = time;
    
    // Note: lf_schedule(start, 0) is commented out in original
    // If uncommented, would cause infinite loop
}

void node_reaction(int in_value, Message* out_msg, int delay) {
    __ESBMC_requires(out_msg != NULL);
    __ESBMC_requires(current_time >= 0);
    __ESBMC_requires(delay >= 0);
    
    __ESBMC_assigns(out_msg->value, out_msg->timestamp, out_msg->valid);
    __ESBMC_ensures(out_msg->value == in_value + 1);
    __ESBMC_ensures(out_msg->timestamp == __ESBMC_old(current_time) + delay);
    __ESBMC_ensures(out_msg->valid == true);
    
    out_msg->value = in_value + 1;
    out_msg->timestamp = current_time + delay;
    out_msg->valid = true;
}

void process_messages_at_time(int time) {
    __ESBMC_requires(time >= 0);
    __ESBMC_requires(time <= MAX_TIME);
    
    /* assigns: all message buffer fields, current_time, source fields */
    __ESBMC_assigns(current_time);
    __ESBMC_assigns(msg_s_to_n1.value, msg_s_to_n1.timestamp, msg_s_to_n1.valid);
    __ESBMC_assigns(msg_n1_to_n2.value, msg_n1_to_n2.timestamp, msg_n1_to_n2.valid);
    __ESBMC_assigns(msg_n2_to_n3.value, msg_n2_to_n3.timestamp, msg_n2_to_n3.valid);
    __ESBMC_assigns(msg_n3_to_n4.value, msg_n3_to_n4.timestamp, msg_n3_to_n4.valid);
    __ESBMC_assigns(msg_n4_to_n5.value, msg_n4_to_n5.timestamp, msg_n4_to_n5.valid);
    __ESBMC_assigns(msg_n5_to_s.value, msg_n5_to_s.timestamp, msg_n5_to_s.valid);
    __ESBMC_assigns(source.received, source.reaction_2_fired, source.reaction_2_fire_time);

    /* (1) current_time is set to the parameter */
    __ESBMC_ensures(current_time == time);

    /* ===== msg_s_to_n1 processing ===== */
    /* If msg_s_to_n1 arrives: consumed, and msg_n1_to_n2 gets new message */
    __ESBMC_ensures(
        !(__ESBMC_old(msg_s_to_n1.valid) & (__ESBMC_old(msg_s_to_n1.timestamp) == time)) |
        ((msg_s_to_n1.valid == 0) &
         (msg_n1_to_n2.value == __ESBMC_old(msg_s_to_n1.value) + 1) &
         (msg_n1_to_n2.timestamp == time + 1) &
         (msg_n1_to_n2.valid == 1))
    );
    /* If msg_s_to_n1 does NOT arrive: msg_s_to_n1 unchanged */
    __ESBMC_ensures(
        (__ESBMC_old(msg_s_to_n1.valid) & (__ESBMC_old(msg_s_to_n1.timestamp) == time)) |
        ((msg_s_to_n1.value == __ESBMC_old(msg_s_to_n1.value)) &
         (msg_s_to_n1.timestamp == __ESBMC_old(msg_s_to_n1.timestamp)) &
         (msg_s_to_n1.valid == __ESBMC_old(msg_s_to_n1.valid)))
    );

    /* ===== msg_n1_to_n2 processing ===== */
    /* Note: only processes if ORIGINALLY valid & timestamp==time (before first if may overwrite it).
       Since new msg_n1_to_n2 from first if has timestamp=time+1≠time, no chain reaction. */
    __ESBMC_ensures(
        !(__ESBMC_old(msg_n1_to_n2.valid) & (__ESBMC_old(msg_n1_to_n2.timestamp) == time)) |
        ((msg_n2_to_n3.value == __ESBMC_old(msg_n1_to_n2.value) + 1) &
         (msg_n2_to_n3.timestamp == time + 1) &
         (msg_n2_to_n3.valid == 1))
    );

    /* ===== msg_n2_to_n3 processing ===== */
    __ESBMC_ensures(
        !(__ESBMC_old(msg_n2_to_n3.valid) & (__ESBMC_old(msg_n2_to_n3.timestamp) == time)) |
        ((msg_n3_to_n4.value == __ESBMC_old(msg_n2_to_n3.value) + 1) &
         (msg_n3_to_n4.timestamp == time + 1) &
         (msg_n3_to_n4.valid == 1))
    );

    /* ===== msg_n3_to_n4 processing ===== */
    __ESBMC_ensures(
        !(__ESBMC_old(msg_n3_to_n4.valid) & (__ESBMC_old(msg_n3_to_n4.timestamp) == time)) |
        ((msg_n4_to_n5.value == __ESBMC_old(msg_n3_to_n4.value) + 1) &
         (msg_n4_to_n5.timestamp == time + 1) &
         (msg_n4_to_n5.valid == 1))
    );

    /* ===== msg_n4_to_n5 processing ===== */
    __ESBMC_ensures(
        !(__ESBMC_old(msg_n4_to_n5.valid) & (__ESBMC_old(msg_n4_to_n5.timestamp) == time)) |
        ((msg_n5_to_s.value == __ESBMC_old(msg_n4_to_n5.value) + 1) &
         (msg_n5_to_s.timestamp == time + 1) &
         (msg_n5_to_s.valid == 1))
    );

    /* ===== msg_n5_to_s processing ===== */
    /* If msg_n5_to_s arrives: source_reaction_in is called */
    __ESBMC_ensures(
        !(__ESBMC_old(msg_n5_to_s.valid) & (__ESBMC_old(msg_n5_to_s.timestamp) == time)) |
        (source.reaction_2_fired == 1 & source.reaction_2_fire_time == time &
         source.received == __ESBMC_old(msg_n5_to_s.value) &
         msg_n5_to_s.valid == 0)
    );

    /* ===== Monotonicity: source.reaction_2_fired can only go from false to true ===== */
    /* If msg_n5_to_s does NOT arrive: source fields preserved */
    __ESBMC_ensures(
        (__ESBMC_old(msg_n5_to_s.valid) & (__ESBMC_old(msg_n5_to_s.timestamp) == time)) |
        ((source.received == __ESBMC_old(source.received)) &
         (source.reaction_2_fired == __ESBMC_old(source.reaction_2_fired)) &
         (source.reaction_2_fire_time == __ESBMC_old(source.reaction_2_fire_time)))
    );

    /* ===== Preservation of non-arriving messages ===== */
    /* If msg_n1_to_n2 was NOT originally arriving AND msg_s_to_n1 was NOT arriving either
       (so no new msg_n1_to_n2 was generated): msg_n1_to_n2 unchanged */
    __ESBMC_ensures(
        (__ESBMC_old(msg_n1_to_n2.valid) & (__ESBMC_old(msg_n1_to_n2.timestamp) == time)) |
        (__ESBMC_old(msg_s_to_n1.valid) & (__ESBMC_old(msg_s_to_n1.timestamp) == time)) |
        ((msg_n1_to_n2.value == __ESBMC_old(msg_n1_to_n2.value)) &
         (msg_n1_to_n2.timestamp == __ESBMC_old(msg_n1_to_n2.timestamp)) &
         (msg_n1_to_n2.valid == __ESBMC_old(msg_n1_to_n2.valid)))
    );
    /* If msg_n2_to_n3 not originally arriving AND msg_n1_to_n2 not arriving: unchanged */
    __ESBMC_ensures(
        (__ESBMC_old(msg_n2_to_n3.valid) & (__ESBMC_old(msg_n2_to_n3.timestamp) == time)) |
        (__ESBMC_old(msg_n1_to_n2.valid) & (__ESBMC_old(msg_n1_to_n2.timestamp) == time)) |
        ((msg_n2_to_n3.value == __ESBMC_old(msg_n2_to_n3.value)) &
         (msg_n2_to_n3.timestamp == __ESBMC_old(msg_n2_to_n3.timestamp)) &
         (msg_n2_to_n3.valid == __ESBMC_old(msg_n2_to_n3.valid)))
    );
    /* msg_n3_to_n4 preserved if neither msg_n3_to_n4 nor msg_n2_to_n3 arrives */
    __ESBMC_ensures(
        (__ESBMC_old(msg_n3_to_n4.valid) & (__ESBMC_old(msg_n3_to_n4.timestamp) == time)) |
        (__ESBMC_old(msg_n2_to_n3.valid) & (__ESBMC_old(msg_n2_to_n3.timestamp) == time)) |
        ((msg_n3_to_n4.value == __ESBMC_old(msg_n3_to_n4.value)) &
         (msg_n3_to_n4.timestamp == __ESBMC_old(msg_n3_to_n4.timestamp)) &
         (msg_n3_to_n4.valid == __ESBMC_old(msg_n3_to_n4.valid)))
    );
    /* msg_n4_to_n5 preserved if neither msg_n4_to_n5 nor msg_n3_to_n4 arrives */
    __ESBMC_ensures(
        (__ESBMC_old(msg_n4_to_n5.valid) & (__ESBMC_old(msg_n4_to_n5.timestamp) == time)) |
        (__ESBMC_old(msg_n3_to_n4.valid) & (__ESBMC_old(msg_n3_to_n4.timestamp) == time)) |
        ((msg_n4_to_n5.value == __ESBMC_old(msg_n4_to_n5.value)) &
         (msg_n4_to_n5.timestamp == __ESBMC_old(msg_n4_to_n5.timestamp)) &
         (msg_n4_to_n5.valid == __ESBMC_old(msg_n4_to_n5.valid)))
    );
    /* msg_n5_to_s preserved if neither msg_n5_to_s nor msg_n4_to_n5 arrives */
    __ESBMC_ensures(
        (__ESBMC_old(msg_n5_to_s.valid) & (__ESBMC_old(msg_n5_to_s.timestamp) == time)) |
        (__ESBMC_old(msg_n4_to_n5.valid) & (__ESBMC_old(msg_n4_to_n5.timestamp) == time)) |
        ((msg_n5_to_s.value == __ESBMC_old(msg_n5_to_s.value)) &
         (msg_n5_to_s.timestamp == __ESBMC_old(msg_n5_to_s.timestamp)) &
         (msg_n5_to_s.valid == __ESBMC_old(msg_n5_to_s.valid)))
    );
    
    current_time = time;
    
    // Check which messages arrive at current time
    if (msg_s_to_n1.valid && msg_s_to_n1.timestamp == time) {
        node_reaction(msg_s_to_n1.value, &msg_n1_to_n2, 1);
        msg_s_to_n1.valid = false;
    }
    
    if (msg_n1_to_n2.valid && msg_n1_to_n2.timestamp == time) {
        node_reaction(msg_n1_to_n2.value, &msg_n2_to_n3, 1);
        msg_n1_to_n2.valid = false;
    }
    
    if (msg_n2_to_n3.valid && msg_n2_to_n3.timestamp == time) {
        node_reaction(msg_n2_to_n3.value, &msg_n3_to_n4, 1);
        msg_n2_to_n3.valid = false;
    }
    
    if (msg_n3_to_n4.valid && msg_n3_to_n4.timestamp == time) {
        node_reaction(msg_n3_to_n4.value, &msg_n4_to_n5, 1);
        msg_n3_to_n4.valid = false;
    }
    
    if (msg_n4_to_n5.valid && msg_n4_to_n5.timestamp == time) {
        node_reaction(msg_n4_to_n5.value, &msg_n5_to_s, 1);
        msg_n4_to_n5.valid = false;
    }
    
    if (msg_n5_to_s.valid && msg_n5_to_s.timestamp == time) {
        source_reaction_in(msg_n5_to_s.value, time);
        msg_n5_to_s.valid = false;
    }
}

int main() {
    // Initialize
    source_startup();
    
    // Time 0: startup and start reaction
    current_time = 0;
    source_reaction_start();
    
    // Simulate time progression from 1 to MAX_TIME
    for (int t = 1; t <= MAX_TIME; t++) {
        process_messages_at_time(t);
    }
    
    // Property: F[0, 10 nsec](Ring_s_reaction_2)
    // "Eventually within [0, 10 nsec], Ring_s_reaction_2 fires"
    // Ring_s_reaction_2 is the source_reaction_in (reaction(in))
    assert(source.reaction_2_fired && source.reaction_2_fire_time <= MAX_TIME);
    
    return 0;
}