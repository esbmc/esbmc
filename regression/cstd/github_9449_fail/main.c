#include <stdlib.h>
#include <string.h>

int main(void)
{
  char *p = malloc(4);
  if (!p)
    return 0;
  memcpy(p, "PING", 4);
  int r = memcmp(p, "HELLO-V1", 8);
  free(p);
  return r != 0;
}
