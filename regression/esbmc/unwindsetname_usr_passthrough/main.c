// Test --unwindsetname with internal USR format passthrough
// This tests that the internal c:@F@func# format is properly supported

#include <assert.h>

void compute_sum() {
  int i, sum = 0;
  for (i = 0; i < 10; i++) {
    sum += i;
  }
  assert(sum == 45);  // 0+1+2+...+9 = 45
}

void compute_product() {
  int j, prod = 1;
  for (j = 0; j < 5; j++) {
    prod *= 2;
  }
  assert(prod == 32);  // 2^5 = 32
}

int main() {
  // Using internal USR format: c:@F@compute_sum#:0:11
  compute_sum();

  // Mix USR format with user-friendly syntax: compute_product:0:6
  compute_product();

  return 0;
}
