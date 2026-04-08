/* Test case for multiple references to the same __ESBMC_old expression
 * This tests that the same __ESBMC_old(expr) can appear multiple times
 * in different ensures clauses without causing segfault.
 * 
 * Before fix: Segmentation fault
 * After fix: Verification successful
 */
#include <stddef.h>

typedef struct {
    int inUse;  // 0 = free, 1 = in use
} State;

// Function that demonstrates multiple __ESBMC_old references
// The same expression __ESBMC_old(s->inUse) appears 3 times
int request(State *s) {
    __ESBMC_requires(s != NULL);
    
    // These ensures clauses reference __ESBMC_old(s->inUse) multiple times
    // If old inUse was 0, then new inUse is 1
    __ESBMC_ensures(__ESBMC_old(s->inUse) != 0 || s->inUse == 1);
    // If old inUse was 0, then return value is 1
    __ESBMC_ensures(__ESBMC_old(s->inUse) != 0 || __ESBMC_return_value == 1);
    // If old inUse was not 0, then return value is 0
    __ESBMC_ensures(__ESBMC_old(s->inUse) == 0 || __ESBMC_return_value == 0);
    
    if (s->inUse == 0) {
        s->inUse = 1;  // Grant request
        return 1;
    }
    return 0;  // Deny request
}

int main() {
    State s = {0};
    int result = request(&s);
    return 0;
}
