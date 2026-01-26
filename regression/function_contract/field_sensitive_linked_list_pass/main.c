/* =============================================================================
 * TEST: Linked List Node Field Isolation — Pillar 1 (Access Path Restoration)
 * =============================================================================
 *
 * PURPOSE:
 *   Demonstrate field-sensitive verification on a linked list node.
 *   Modifying the data field of a node should NOT affect its next pointer,
 *   and modifying the next pointer should NOT affect the data.
 *   This is critical for verifying data structure invariants modularly.
 *
 * TECHNICAL CHALLENGE:
 *   Pointer fields in structs create aliasing complexity. The access path
 *   p->data vs p->next must be distinguished even though both are accessed
 *   through the same base pointer.
 *
 * REAL-WORLD RELEVANCE:
 *   Kernel linked lists (Linux list_head), network buffer chains, and
 *   message queue implementations all require field-level isolation to
 *   avoid false positives when verifying list operations.
 *
 * EXPECTED: VERIFICATION SUCCESSFUL
 * =========================================================================== */

#include <assert.h>
#include <stddef.h>

typedef struct Node {
    int data;
    struct Node *next;
} Node;

/* Update only the data field, preserving the link structure */
void node_set_data(Node *n, int val) {
    __ESBMC_requires(n != NULL);
    __ESBMC_assigns(n->data);
    __ESBMC_ensures(n->data == val);
    __ESBMC_ensures(n->next == __ESBMC_old(n->next));

    n->data = val;
}

int main() {
    Node second;
    second.data = 200;
    second.next = NULL;

    Node first;
    first.data = 100;
    first.next = &second;

    node_set_data(&first, 999);

    assert(first.data == 999);         /* Modified */
    assert(first.next == &second);     /* Link preserved */
    assert(first.next->data == 200);   /* Second node untouched */

    return 0;
}
