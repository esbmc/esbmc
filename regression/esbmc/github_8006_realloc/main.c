// #8006
#include <stdlib.h>
int main()
{
  int *p = malloc(sizeof(int));
  if (!p) return 0;
  *p = 1;
  int a = *p + 1;
  int *q = realloc(p, 100 * sizeof(int));
  int b = *p + 1; // UAF when realloc moved/freed
  return a + b;
}
