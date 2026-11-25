/* #9221: no address is read, so no object non-overlap constraint is emitted. */
#include <stdlib.h>
int main(void)
{
  for (int i = 0; i < 20; i++)
  {
    int *p = malloc(sizeof(int));
    if (p)
      *p = i;
    free(p);
  }
  return 0;
}
