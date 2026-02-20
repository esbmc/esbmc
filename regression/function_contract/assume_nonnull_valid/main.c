/* Test case for --assume-nonnull-valid option
 * This tests that pointer parameters are assumed to be valid objects
 * when using --enforce-contract with --assume-nonnull-valid
 */
#include <assert.h>
#include <stddef.h>

typedef struct {
    int x;
} S;

void f(S *p) {
    __ESBMC_requires(p != NULL);
    __ESBMC_ensures(p->x == 10);
    p->x = 10;
}

int main() {
    S s;
    f(&s);
    assert(s.x == 10);
    return 0;
}
