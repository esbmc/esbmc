// Test multiple loops in single function with different unwind bounds
// Each loop needs a different number of unwinds

#include <assert.h>

void multi_loop() {
  int i, j;
  int sum = 0;
  int prod = 1;
  int limit1, limit2;

  __ESBMC_assume(limit1 == 5);
  __ESBMC_assume(limit2 == 3);

  // First loop (index 0) - needs 5 unwinds
  for (i = 0; i < limit1; i++) {
    sum += i;
  }

  // Second loop (index 1) - needs 3 unwinds
  for (j = 0; j < limit2; j++) {
    prod *= 2;
  }

  assert(sum == 10);   // 0+1+2+3+4 = 10
  assert(prod == 8);   // 2^3 = 8
}

int main() {
  multi_loop();
  return 0;
}
