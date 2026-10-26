// #8006
#include <stdlib.h>
int main()
{
  int *p = malloc(sizeof(int));
  int *r = malloc(sizeof(int));
  if (!p || !r) return 0;
  *p = 1;
  *r = 2;
  int a = *p + 1;
  free(r);
  int b = *p + 1;
  free(p);
  return a + b;
}
