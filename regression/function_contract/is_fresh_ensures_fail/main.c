/* is_fresh_ensures_fail: Test __ESBMC_is_fresh in ensures clause with non-fresh memory
 * Tests that ensures clause with __ESBMC_is_fresh correctly detects when pointer
 * does not point to fresh memory (e.g., stack variable)
 * This test should FAIL because we pass a stack variable instead of fresh memory
 */
#include <stdbool.h>
#include <stdlib.h>

typedef struct {
    int x;
    int y;
} Point;

// Function that should initialize a fresh Point
void init_point(Point* p, int x, int y) {
    __ESBMC_requires(p != NULL);
    __ESBMC_requires(x >= 0);
    __ESBMC_requires(y >= 0);
    
    // Ensures that p points to fresh memory
    __ESBMC_ensures(__ESBMC_is_fresh(&p, sizeof(Point)));
    __ESBMC_ensures(p->x == x);
    __ESBMC_ensures(p->y == y);
    
    // Initialize the point
    p->x = x;
    p->y = y;
}

int main() {
    // WRONG: Pass a stack variable instead of fresh memory
    // This should trigger the ensures failure because stack_point is not fresh
    Point stack_point;
    // The ensures clause will verify that &stack_point points to fresh memory
    // Since stack_point is a stack variable (not fresh), this should FAIL
    init_point(&stack_point, 10, 20);  // This should fail ensures check!
    
    return 0;
}

