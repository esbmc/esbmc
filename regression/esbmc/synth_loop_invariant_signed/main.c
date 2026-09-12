/* A signed counter is handled when no accumulator multiplies by a symbolic
 * value: the third disjunct i == i0 makes establishment unconditional, which
 * is what a possibly-negative bound needs. Plain BMC only reports the
 * unwinding assertion here, so the invariant is what proves it for all n. */
#include <assert.h>
int main(void) { int n; int i = 0; while (i < n) i++; assert(i == n || n <= 0); return 0; }
