/*
 * Enforce process_messages_at_time contract with leaf functions replaced.
 *
 * Verifies that the implementation of process_messages_at_time satisfies
 * its contract for ALL possible global states (nondet harness).
 *
 * Command:
 *   esbmc main.c --enforce-contract process_messages_at_time \
 *     --replace-call-with-contract "source_reaction_in,node_reaction" \
 *     --no-align-check
 */

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>

#define MAX_TIME 10

int nondet_int();

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

int current_time = 0;
Source source;
Node n1, n2, n3, n4, n5;

Message msg_s_to_n1 = {0, -1, false};
Message msg_n1_to_n2 = {0, -1, false};
Message msg_n2_to_n3 = {0, -1, false};
Message msg_n3_to_n4 = {0, -1, false};
Message msg_n4_to_n5 = {0, -1, false};
Message msg_n5_to_s = {0, -1, false};

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

    __ESBMC_ensures(current_time == time);

    /* Single ternary ensures to avoid duplicating old() references
     * (multiple old() on same variable across ensures can cause index mismatch) */
    __ESBMC_ensures(
        __ESBMC_old(msg_n5_to_s.valid) && __ESBMC_old(msg_n5_to_s.timestamp) == time
        ? (source.reaction_2_fired == true && source.reaction_2_fire_time == time)
        : (source.reaction_2_fired == __ESBMC_old(source.reaction_2_fired) &&
           source.reaction_2_fire_time == __ESBMC_old(source.reaction_2_fire_time) &&
           source.received == __ESBMC_old(source.received)));

    current_time = time;

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
    /* Nondet harness: universal contract verification */
    msg_s_to_n1.valid = nondet_int() != 0;
    msg_s_to_n1.value = nondet_int();
    msg_s_to_n1.timestamp = nondet_int();

    msg_n1_to_n2.valid = nondet_int() != 0;
    msg_n1_to_n2.value = nondet_int();
    msg_n1_to_n2.timestamp = nondet_int();

    msg_n2_to_n3.valid = nondet_int() != 0;
    msg_n2_to_n3.value = nondet_int();
    msg_n2_to_n3.timestamp = nondet_int();

    msg_n3_to_n4.valid = nondet_int() != 0;
    msg_n3_to_n4.value = nondet_int();
    msg_n3_to_n4.timestamp = nondet_int();

    msg_n4_to_n5.valid = nondet_int() != 0;
    msg_n4_to_n5.value = nondet_int();
    msg_n4_to_n5.timestamp = nondet_int();

    msg_n5_to_s.valid = nondet_int() != 0;
    msg_n5_to_s.value = nondet_int();
    msg_n5_to_s.timestamp = nondet_int();

    source.received = nondet_int();
    source.reaction_2_fired = nondet_int() != 0;
    source.reaction_2_fire_time = nondet_int();

    int time = nondet_int();
    __ESBMC_assume(time >= 0 && time <= MAX_TIME);

    process_messages_at_time(time);

    return 0;
}
