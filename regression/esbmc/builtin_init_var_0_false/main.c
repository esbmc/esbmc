#include <assert.h>

int main() {
  int v;
  v = 0;

  assert(v == 0);
  __ESBMC_init_var(&v);
  assert(v == 0);
}
