#include <assert.h>

int main() {
  int x = nondet_int();
  assert((x | ~x) != -1);
  return 0;
}

