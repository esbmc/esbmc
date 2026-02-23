#include <assert.h>

int main() {
  int x = nondet_int();
  assert((x ^ x) != 0);
  return 0;
}

