// Test replace-call-with-contract mode with __ESBMC_and, __ESBMC_or macros
// to avoid short-circuit evaluation issues with __ESBMC_old() in ensures.
//
// This test verifies that complex disjunctive ensures clauses with __ESBMC_old
// work correctly when the function call is replaced with its contract.

#include <assert.h>

typedef struct {
    int id;
    int value;
    int updated;
} Node;

void process(Node* node, int input) {
    __ESBMC_requires(node != 0);
    __ESBMC_assigns(node->value, node->updated);
    // Three-way disjunction using helper macros:
    // Case 1: input > node->id -> update value, mark as updated
    // Case 2: input == node->id -> just mark as updated  
    // Case 3: input < node->id -> no change
    __ESBMC_ensures(
        __ESBMC_or(
            __ESBMC_or(
                __ESBMC_and(__ESBMC_and(input > node->id, node->value == input), node->updated == 1),
                __ESBMC_and(__ESBMC_and(input == node->id, node->value == __ESBMC_old(node->value)), node->updated == 1)
            ),
            __ESBMC_and(__ESBMC_and(input < node->id, node->value == __ESBMC_old(node->value)), node->updated == __ESBMC_old(node->updated))
        )
    );
    
    if (input > node->id) {
        node->value = input;
        node->updated = 1;
    } else if (input == node->id) {
        node->updated = 1;
    }
    // input < node->id: no change
}

int main() {
    Node n = {5, 0, 0};
    
    // Test case 1: input > id
    process(&n, 10);
    assert(n.value == 10);
    assert(n.updated == 1);
    
    // Reset
    n.value = 0;
    n.updated = 0;
    
    // Test case 2: input == id
    process(&n, 5);
    assert(n.value == 0);  // unchanged
    assert(n.updated == 1);
    
    // Reset
    n.updated = 0;
    
    // Test case 3: input < id
    process(&n, 2);
    assert(n.value == 0);  // unchanged
    assert(n.updated == 0);  // unchanged
    
    return 0;
}
