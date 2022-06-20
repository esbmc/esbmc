#include <assert.h>

struct obj {
  int a;
  int b;
};

int main() {
  struct obj O;
  O.a = 4;
  assert(O.a == 4);
  __ESBMC_init_object(&O);
  assert(O.a == 4);
}
