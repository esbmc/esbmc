// Test --unwindsetname with multiple functions
// Each function needs different unwind bounds

#include <assert.h>

void func_ten() {
  int i, limit;
  int sum = 0;

  __ESBMC_assume(limit == 10);

  for (i = 0; i < limit; i++) {
    sum += i;
  }

  assert(sum == 45);
}

void func_five() {
  int j, limit;
  int prod = 1;

  __ESBMC_assume(limit == 5);

  for (j = 0; j < limit; j++) {
    prod *= 2;
  }

  assert(prod == 32);
}

void func_three() {
  int k, limit;
  int count = 0;

  __ESBMC_assume(limit == 3);

  for (k = 0; k < limit; k++) {
    count++;
  }

  assert(count == 3);
}

int main() {
  func_ten();
  func_five();
  func_three();
  return 0;
}
