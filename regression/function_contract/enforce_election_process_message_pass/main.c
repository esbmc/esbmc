#include <assert.h>
#include <stddef.h>

typedef struct {
    int id;
    int elected;
    int pending_value;
    int pending_time;
    int has_pending;
} Node;

void init_nodes(Node* nodes) {
    __ESBMC_requires(nodes != ((void*)0));

    __ESBMC_assigns(nodes[0].id, nodes[0].elected, nodes[0].pending_value, nodes[0].pending_time, nodes[0].has_pending);
    __ESBMC_assigns(nodes[1].id, nodes[1].elected, nodes[1].pending_value, nodes[1].pending_time, nodes[1].has_pending);
    __ESBMC_assigns(nodes[2].id, nodes[2].elected, nodes[2].pending_value, nodes[2].pending_time, nodes[2].has_pending);
    __ESBMC_ensures(nodes[0].id == 0 && nodes[0].elected == 0 && nodes[0].pending_value == -1 && nodes[0].pending_time == -1 && nodes[0].has_pending == 0);
    __ESBMC_ensures(nodes[1].id == 1 && nodes[1].elected == 0 && nodes[1].pending_value == -1 && nodes[1].pending_time == -1 && nodes[1].has_pending == 0);
    __ESBMC_ensures(nodes[2].id == 2 && nodes[2].elected == 0 && nodes[2].pending_value == -1 && nodes[2].pending_time == -1 && nodes[2].has_pending == 0);

    for (int i = 0; i < 3; i++) {
        nodes[i].id = i;
        nodes[i].elected = 0;
        nodes[i].pending_value = -1;
        nodes[i].pending_time = -1;
        nodes[i].has_pending = 0;
    }
}

void process_message(Node* node, int value) {
    __ESBMC_requires(node != ((void*)0));
    __ESBMC_assigns(node->pending_value, node->pending_time, node->has_pending, node->elected);
    __ESBMC_ensures(
        ((value > node->id) & (node->pending_value == value) & (node->pending_time == 10) & (node->has_pending == 1) & (node->elected == __ESBMC_old(node->elected))) |
        ((value == node->id) & (node->elected == 1) & (node->pending_value == __ESBMC_old(node->pending_value)) & (node->pending_time == __ESBMC_old(node->pending_time)) & (node->has_pending == __ESBMC_old(node->has_pending))) |
        ((value < node->id) & (node->elected == __ESBMC_old(node->elected)) & (node->pending_value == __ESBMC_old(node->pending_value)) & (node->pending_time == __ESBMC_old(node->pending_time)) & (node->has_pending == __ESBMC_old(node->has_pending)))
    );

    if (value > node->id) {
        node->pending_value = value;
        node->pending_time = 10;
        node->has_pending = 1;
    } else if (value == node->id) {
        node->elected = 1;
    }
}

void send_to_next(Node* nodes, int from_idx, int value) {
    __ESBMC_requires(nodes != ((void*)0));
    __ESBMC_requires(from_idx >= 0 && from_idx < 3);

    __ESBMC_assigns(nodes[0].pending_value, nodes[0].pending_time, nodes[0].has_pending, nodes[0].elected);
    __ESBMC_assigns(nodes[1].pending_value, nodes[1].pending_time, nodes[1].has_pending, nodes[1].elected);
    __ESBMC_assigns(nodes[2].pending_value, nodes[2].pending_time, nodes[2].has_pending, nodes[2].elected);

    /* IDs never change */
    __ESBMC_ensures(nodes[0].id == __ESBMC_old(nodes[0].id));
    __ESBMC_ensures(nodes[1].id == __ESBMC_old(nodes[1].id));
    __ESBMC_ensures(nodes[2].id == __ESBMC_old(nodes[2].id));

    /* Case from_idx == 0: target is nodes[1], unchanged: nodes[0], nodes[2] */
    __ESBMC_ensures((from_idx != 0) | (
        (nodes[0].pending_value == __ESBMC_old(nodes[0].pending_value)) &
        (nodes[0].pending_time == __ESBMC_old(nodes[0].pending_time)) &
        (nodes[0].has_pending == __ESBMC_old(nodes[0].has_pending)) &
        (nodes[0].elected == __ESBMC_old(nodes[0].elected)) &
        (nodes[2].pending_value == __ESBMC_old(nodes[2].pending_value)) &
        (nodes[2].pending_time == __ESBMC_old(nodes[2].pending_time)) &
        (nodes[2].has_pending == __ESBMC_old(nodes[2].has_pending)) &
        (nodes[2].elected == __ESBMC_old(nodes[2].elected))
    ));
    __ESBMC_ensures((from_idx != 0) | (
        ((value > __ESBMC_old(nodes[1].id)) & (nodes[1].pending_value == value) & (nodes[1].pending_time == 10) & (nodes[1].has_pending == 1) & (nodes[1].elected == __ESBMC_old(nodes[1].elected))) |
        ((value == __ESBMC_old(nodes[1].id)) & (nodes[1].elected == 1) & (nodes[1].pending_value == __ESBMC_old(nodes[1].pending_value)) & (nodes[1].pending_time == __ESBMC_old(nodes[1].pending_time)) & (nodes[1].has_pending == __ESBMC_old(nodes[1].has_pending))) |
        ((value < __ESBMC_old(nodes[1].id)) & (nodes[1].elected == __ESBMC_old(nodes[1].elected)) & (nodes[1].pending_value == __ESBMC_old(nodes[1].pending_value)) & (nodes[1].pending_time == __ESBMC_old(nodes[1].pending_time)) & (nodes[1].has_pending == __ESBMC_old(nodes[1].has_pending)))
    ));

    /* Case from_idx == 1: target is nodes[2], unchanged: nodes[0], nodes[1] */
    __ESBMC_ensures((from_idx != 1) | (
        (nodes[0].pending_value == __ESBMC_old(nodes[0].pending_value)) &
        (nodes[0].pending_time == __ESBMC_old(nodes[0].pending_time)) &
        (nodes[0].has_pending == __ESBMC_old(nodes[0].has_pending)) &
        (nodes[0].elected == __ESBMC_old(nodes[0].elected)) &
        (nodes[1].pending_value == __ESBMC_old(nodes[1].pending_value)) &
        (nodes[1].pending_time == __ESBMC_old(nodes[1].pending_time)) &
        (nodes[1].has_pending == __ESBMC_old(nodes[1].has_pending)) &
        (nodes[1].elected == __ESBMC_old(nodes[1].elected))
    ));
    __ESBMC_ensures((from_idx != 1) | (
        ((value > __ESBMC_old(nodes[2].id)) & (nodes[2].pending_value == value) & (nodes[2].pending_time == 10) & (nodes[2].has_pending == 1) & (nodes[2].elected == __ESBMC_old(nodes[2].elected))) |
        ((value == __ESBMC_old(nodes[2].id)) & (nodes[2].elected == 1) & (nodes[2].pending_value == __ESBMC_old(nodes[2].pending_value)) & (nodes[2].pending_time == __ESBMC_old(nodes[2].pending_time)) & (nodes[2].has_pending == __ESBMC_old(nodes[2].has_pending))) |
        ((value < __ESBMC_old(nodes[2].id)) & (nodes[2].elected == __ESBMC_old(nodes[2].elected)) & (nodes[2].pending_value == __ESBMC_old(nodes[2].pending_value)) & (nodes[2].pending_time == __ESBMC_old(nodes[2].pending_time)) & (nodes[2].has_pending == __ESBMC_old(nodes[2].has_pending)))
    ));

    /* Case from_idx == 2: target is nodes[0], unchanged: nodes[1], nodes[2] */
    __ESBMC_ensures((from_idx != 2) | (
        (nodes[1].pending_value == __ESBMC_old(nodes[1].pending_value)) &
        (nodes[1].pending_time == __ESBMC_old(nodes[1].pending_time)) &
        (nodes[1].has_pending == __ESBMC_old(nodes[1].has_pending)) &
        (nodes[1].elected == __ESBMC_old(nodes[1].elected)) &
        (nodes[2].pending_value == __ESBMC_old(nodes[2].pending_value)) &
        (nodes[2].pending_time == __ESBMC_old(nodes[2].pending_time)) &
        (nodes[2].has_pending == __ESBMC_old(nodes[2].has_pending)) &
        (nodes[2].elected == __ESBMC_old(nodes[2].elected))
    ));
    __ESBMC_ensures((from_idx != 2) | (
        ((value > __ESBMC_old(nodes[0].id)) & (nodes[0].pending_value == value) & (nodes[0].pending_time == 10) & (nodes[0].has_pending == 1) & (nodes[0].elected == __ESBMC_old(nodes[0].elected))) |
        ((value == __ESBMC_old(nodes[0].id)) & (nodes[0].elected == 1) & (nodes[0].pending_value == __ESBMC_old(nodes[0].pending_value)) & (nodes[0].pending_time == __ESBMC_old(nodes[0].pending_time)) & (nodes[0].has_pending == __ESBMC_old(nodes[0].has_pending))) |
        ((value < __ESBMC_old(nodes[0].id)) & (nodes[0].elected == __ESBMC_old(nodes[0].elected)) & (nodes[0].pending_value == __ESBMC_old(nodes[0].pending_value)) & (nodes[0].pending_time == __ESBMC_old(nodes[0].pending_time)) & (nodes[0].has_pending == __ESBMC_old(nodes[0].has_pending)))
    ));

    int next_idx = (from_idx + 1) % 3;
    process_message(&nodes[next_idx], value);
}

int main() {
    Node nodes[3];
    init_nodes(nodes);

    // Time 0: Startup
    send_to_next(nodes, 0, 0);
    send_to_next(nodes, 1, 1);
    send_to_next(nodes, 2, 2);

    // Simulate time progression
    for (int time = 1; time <= 20; time++) {
        for (int i = 0; i < 3; i++) {
            if (nodes[i].has_pending && nodes[i].pending_time == time) {
                send_to_next(nodes, i, nodes[i].pending_value);
                nodes[i].has_pending = 0;
            }
        }
    }

    int sum_elected = nodes[0].elected + nodes[1].elected + nodes[2].elected;
    assert(sum_elected == 1);

    return 0;
}
