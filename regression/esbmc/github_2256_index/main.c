/* #2256: an indexed read of a malloc'd buffer whose length bounds the loop. */
#include <assert.h>
#include <stdlib.h>

int main()
{
  char *p = malloc(4);
  if (!p)
    return 0;
  for (int i = 0; i < 3; i++)
    p[i] = 'a' + i;
  p[3] = 0;
  int n = 0;
  while (p[n])
    n++;
  assert(n == 3);
  return 0;
}
