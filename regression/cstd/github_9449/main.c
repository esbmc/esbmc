#include <assert.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
  char a[4] = {'P', 'I', 'N', 'G'};
  char b[4] = {'H', 'E', 'L', 'L'};
  assert(memcmp(a, b, 4) > 0);
  assert(memcmp(a, a, 4) == 0);
  assert(memcmp(a, b, 0) == 0);

  char *p = malloc(4);
  if (!p)
    return 0;
  memcpy(p, "PING", 4);
  assert(memcmp(p, "PING", 4) == 0);
  assert(memcmp(p, "HELLO-V1", 4) != 0);
  free(p);
  return 0;
}
