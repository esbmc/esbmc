#include <stdlib.h>
int *a, *b;
int n;
#define BLOCK_SIZE 128
void foo()
{
  int i;
  for(i = 0; i < n; i++)
    a[i] = -1;
    for (i = 0; i < BLOCK_SIZE-1; i++)
      b[i] = -1;
}
main()
{
  n = BLOCK_SIZE;
  a = malloc(n * sizeof(*a));
    b = malloc(n * sizeof(*b));
  b *b++ = 0;
  foo();
  if(b[-1])
  {
    free(a);
    free(b);
  }
  else
  {
    free(a);
    free(b);
  }
  return 0;
}
