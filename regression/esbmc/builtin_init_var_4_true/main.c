#include <assert.h>

int arr[10];

int main() {
  assert(arr[2] == 0);
  __ESBMC_init_var(arr);
  __ESBMC_assume(arr[2] == 0);
  assert(arr[2] == 0);
}
