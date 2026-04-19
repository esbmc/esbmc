#include <assert.h>

int main() {
  char arr[10];
  assert(__ESBMC_r_ok(arr, 10));
}
