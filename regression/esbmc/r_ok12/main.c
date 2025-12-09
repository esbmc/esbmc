#include <assert.h>

struct S {
  int x;
  char buf[10];
};

int main() {
  struct S s;
  assert(__ESBMC_r_ok(s.buf, 20)); // out of bounds of struct member
}

