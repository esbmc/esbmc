// Test that --unwindset overrides --unwindsetname
// Function bound says 3, but loop-specific bound says 10

#include <assert.h>

void test_priority() {
  int i, limit;
  int sum = 0;

  __ESBMC_assume(limit == 10);

  // This is loop 1
  for (i = 0; i < limit; i++) {
    sum += i;
  }

  assert(sum == 45);  // Requires 10 unwinds, not 3
}

int main() {
  test_priority();
  return 0;
}
