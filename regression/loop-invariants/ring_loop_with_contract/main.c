/*
 * Ring_loop.c - Full compositional verification
 *
 * Ring topology: source -> n1 -> n2 -> n3 -> n4 -> n5 -> source
 * Messages propagate with 1 nsec delay per hop.
 *
 * ALL functions have contracts and are replaced at call sites.
 * Main loop uses k-induction with loop invariants.
 *
 * Command:
 *   esbmc main.c --loop-invariant \
 *     --replace-call-with-contract "source_startup,source_reaction_start,source_reaction_in,node_reaction,process_messages_at_time" \
 *     --no-align-check
 */

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

/* ============================================================
 * Leaf functions with contracts (replaced at call sites)
 * ============================================================ */

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

    msg_s_to_n1.value = source.received;
    msg_s_to_n1.timestamp = current_time + 1;
    msg_s_to_n1.valid = true;
}

void source_reaction_in(int in_value, int time) {
    __ESBMC_requires(time >= 0);
    __ESBMC_requires(time <= MAX_TIME);

    __ESBMC_assigns(source.received, source.reaction_2_fired, source.reaction_2_fire_time);
    __ESBMC_ensures(source.received == in_value);
    __ESBMC_ensures(source.reaction_2_fired == true);
    __ESBMC_ensures(source.reaction_2_fire_time == time);

    source.received = in_value;
    source.reaction_2_fired = true;
    source.reaction_2_fire_time = time;
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

/* ============================================================
 * process_messages_at_time: with contract for compositional use
 *
 * IFs processed in reverse chain order to avoid read-after-write
 * conflicts (each buffer is read before any reaction writes to it).
 * ============================================================ */
void process_messages_at_time(int time) {
    __ESBMC_requires(time >= 0);
    __ESBMC_requires(time <= MAX_TIME);

    __ESBMC_assigns(current_time);
    __ESBMC_assigns(msg_s_to_n1.valid);
    __ESBMC_assigns(msg_n1_to_n2.value, msg_n1_to_n2.timestamp, msg_n1_to_n2.valid);
    __ESBMC_assigns(msg_n2_to_n3.value, msg_n2_to_n3.timestamp, msg_n2_to_n3.valid);
    __ESBMC_assigns(msg_n3_to_n4.value, msg_n3_to_n4.timestamp, msg_n3_to_n4.valid);
    __ESBMC_assigns(msg_n4_to_n5.value, msg_n4_to_n5.timestamp, msg_n4_to_n5.valid);
    __ESBMC_assigns(msg_n5_to_s.value, msg_n5_to_s.timestamp, msg_n5_to_s.valid);
    __ESBMC_assigns(source.received, source.reaction_2_fired, source.reaction_2_fire_time);

    /* Postcondition 1: current_time is set to the given time */
    __ESBMC_ensures(current_time == time);

    /* Postcondition 2+3: combined as ternary to avoid duplicate old() references
     * (ESBMC old() snapshot indexing can misalign across multiple ensures) */
    __ESBMC_ensures(
        __ESBMC_old(msg_n5_to_s.valid) && __ESBMC_old(msg_n5_to_s.timestamp) == time
        ? (source.reaction_2_fired == true && source.reaction_2_fire_time == time)
        : (source.reaction_2_fired == __ESBMC_old(source.reaction_2_fired) &&
           source.reaction_2_fire_time == __ESBMC_old(source.reaction_2_fire_time) &&
           source.received == __ESBMC_old(source.received)));

    current_time = time;

    /* Process in reverse chain order: read msg_n5_to_s before node_reaction
     * can overwrite it, etc. */
    if (msg_n5_to_s.valid && msg_n5_to_s.timestamp == time) {
        source_reaction_in(msg_n5_to_s.value, time);
        msg_n5_to_s.valid = false;
    }

    if (msg_n4_to_n5.valid && msg_n4_to_n5.timestamp == time) {
        node_reaction(msg_n4_to_n5.value, &msg_n5_to_s, 1);
        msg_n4_to_n5.valid = false;
    }

    if (msg_n3_to_n4.valid && msg_n3_to_n4.timestamp == time) {
        node_reaction(msg_n3_to_n4.value, &msg_n4_to_n5, 1);
        msg_n3_to_n4.valid = false;
    }

    if (msg_n2_to_n3.valid && msg_n2_to_n3.timestamp == time) {
        node_reaction(msg_n2_to_n3.value, &msg_n3_to_n4, 1);
        msg_n2_to_n3.valid = false;
    }

    if (msg_n1_to_n2.valid && msg_n1_to_n2.timestamp == time) {
        node_reaction(msg_n1_to_n2.value, &msg_n2_to_n3, 1);
        msg_n1_to_n2.valid = false;
    }

    if (msg_s_to_n1.valid && msg_s_to_n1.timestamp == time) {
        node_reaction(msg_s_to_n1.value, &msg_n1_to_n2, 1);
        msg_s_to_n1.valid = false;
    }
}

int main() {
    // Initialize
    source_startup();

    // Time 0: startup and start reaction
    current_time = 0;
    source_reaction_start();

    int t;

    /* Loop invariants (safety properties, inductively provable):
     *
     * 1. Loop counter bounds
     * 2. current_time within bounds
     * 3. If reaction_2 fired, it was at a valid time
     */
    __ESBMC_loop_invariant(t >= 1 && t <= MAX_TIME + 1);
    __ESBMC_loop_invariant(current_time >= 0 && current_time <= MAX_TIME);
    __ESBMC_loop_invariant(!source.reaction_2_fired ||
                           (source.reaction_2_fire_time >= 1 &&
                            source.reaction_2_fire_time <= MAX_TIME));
    for (t = 1; t <= MAX_TIME; t++) {
        process_messages_at_time(t);
    }

    /* Post-loop assertions */
    assert(t == MAX_TIME + 1);
    assert(current_time >= 0 && current_time <= MAX_TIME);
    assert(!source.reaction_2_fired ||
           (source.reaction_2_fire_time >= 1 &&
            source.reaction_2_fire_time <= MAX_TIME));

    return 0;
}
