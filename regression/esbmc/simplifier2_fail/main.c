#include <assert.h>

int main() {
  unsigned u = nondet_uint();
  assert(u < 0);
  return 0;
}

