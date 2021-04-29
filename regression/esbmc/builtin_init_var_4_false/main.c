#include <assert.h>

int arr[10];

int main() {
  assert(arr[2] == 0);
  __ESBMC_init_var(arr);
  assert(arr[2] == 0);
}
