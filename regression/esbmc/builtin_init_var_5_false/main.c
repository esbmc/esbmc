#include <assert.h>

struct obj {
  int a;
  int b;
};

struct obj arr[10];

int main() {
  assert(arr[2].b == 0);
  __ESBMC_init_var(arr);
  assert(arr[2].b == 0);
}
