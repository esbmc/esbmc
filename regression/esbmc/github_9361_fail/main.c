// github #9361: the overflowing calloc path is reachable and yields NULL.
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

int main(void)
{
  size_t n = nondet_bool() ? 4 : SIZE_MAX / 2;
  int *p = calloc(n, sizeof(int));
  assert(n == 4);
  free(p);
  return 0;
}
