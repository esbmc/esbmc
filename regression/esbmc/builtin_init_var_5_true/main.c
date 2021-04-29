#include <assert.h>

struct obj {
  int a;
  int b;
};

struct obj arr[10];

int main() {
  assert(arr[2].b == 0);
  __ESBMC_init_var(arr);
  __ESBMC_assume(arr[2].b == 0);
  assert(arr[2].b == 0);
}
