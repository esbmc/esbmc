#include <string.h>

int main() {
  char *src = "Hello";
  char *dest = NULL;
  int count = nondet_int();
  __ESBMC_assume(count <= 5 && 0 < count);
  memcpy(dest, src, count);
  return 0;
}

