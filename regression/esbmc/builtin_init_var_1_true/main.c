#include <assert.h>

int main() {
  int v[10];
  v[9] = 5;

  assert(v[9] == 5);
  __ESBMC_init_var(v);
  __ESBMC_assume(v[9] == 5);
  assert(v[9] == 5);
}
