// Test basic --unwindsetname functionality
// Should verify successfully when function unwound correctly

#include <assert.h>

void compute_sum() {
  int i;
  int sum = 0;
  int limit;

  __ESBMC_assume(limit == 10);

  for (i = 0; i < limit; i++) {
    sum += i;
  }

  assert(sum == 45);  // 0+1+2+...+9 = 45
}

int main() {
  compute_sum();
  return 0;
}
