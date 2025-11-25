// Test --unwindsetname with static (file-local) functions
// Static function name includes filename in mangled form

#include <assert.h>

static void helper() {
  int i, limit;
  int sum = 0;

  __ESBMC_assume(limit == 7);

  for (i = 0; i < limit; i++) {
    sum += i;
  }

  assert(sum == 21);  // 0+1+2+3+4+5+6 = 21
}

int main() {
  helper();
  return 0;
}
