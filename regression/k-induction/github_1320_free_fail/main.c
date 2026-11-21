/* free changes memory state no loop variable holds, so the inductive step
   must still follow the paths the bound cuts out of the loop
   (esbmc/esbmc#1320). */
#include <stdlib.h>
int main()
{
  int *p = malloc(sizeof(int));
  unsigned i = 0;
  while (i < 20)
  {
    if (i == 0)
      free(p);
    i++;
  }
  *p = 1;
  return 0;
}
