// #8006
#include <stdlib.h>
int main()
{
  int *p = malloc(sizeof(int));
  if (!p) return 0;
  int *q = p;
  *p = 1;
  int a = *p + 1;
  free(q);
  int b = *p + 1;
  return a + b;
}
