#include <assert.h>

struct obj {
  int a;
  int b;
};

int main() {
  struct obj O[10];
  O[2].a = 4;
  assert(O[2].a == 4);
  __ESBMC_init_var(O);
  __ESBMC_assume(O[2].a == 4);
  assert(O[2].a == 4);
}
