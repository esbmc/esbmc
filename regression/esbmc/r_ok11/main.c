#include <assert.h>

struct S {
  int x;
  char buf[20];
};

int main() {
  struct S s;
  assert(__ESBMC_r_ok(s.buf, 10)); // within bounds of struct member
}
