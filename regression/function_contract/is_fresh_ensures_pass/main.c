/* is_fresh_ensures_pass: Test __ESBMC_is_fresh in ensures clause with fresh memory
 * Tests that ensures clause with __ESBMC_is_fresh correctly passes when pointer
 * points to fresh memory (malloc allocated)
 * This test should PASS because we pass fresh memory allocated by malloc
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
    // CORRECT: Allocate fresh memory using malloc
    // This should pass the ensures check because malloc allocates fresh memory
    Point* p = malloc(sizeof(Point));
    if (p == NULL) {
        return 1;
    }
    
    // The ensures clause will verify that p points to fresh memory
    // Since we allocated it with malloc, this should pass
    init_point(p, 10, 20);  // This should pass ensures check!
    
    free(p);
    return 0;
}

