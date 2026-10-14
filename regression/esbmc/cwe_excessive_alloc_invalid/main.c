#include <stdlib.h>

int main(void)
{
  char *p = malloc(8);
  free(p);
  return 0;
}
