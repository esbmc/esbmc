#include <string.h>
#include <stdlib.h>
int main() {
  char *p = malloc(6); for (int i = 0; i < 5; i++) p[i] = nondet_char(); p[5] = 0; p[0] = 'x';
  char *e = strchr(p, '.');
  __ESBMC_assert(e == NULL || (*e == '.' && e > p), "dot");
}
