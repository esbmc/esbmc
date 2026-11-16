// github #9361: calloc returns NULL when nmemb * size wraps size_t.
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

int main(void)
{
  size_t n = nondet_bool() ? 4 : SIZE_MAX / 2;
  int *p = calloc(n, sizeof(int));
  if (n != 4)
    assert(p == NULL);
  else if (p)
    assert(p[3] == 0);
  free(p);
  return 0;
}
