#include <string.h>

int main() {
  char *src = NULL;
  char dest[10];
  int count = nondet_int();
  __ESBMC__assume(count <= 5 && 0 < count);
  memcpy(dest, src, count);
  return 0;
}

