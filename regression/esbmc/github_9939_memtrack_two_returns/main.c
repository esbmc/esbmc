/* #9939: a branch with no recorded direction leads to several returns, so the target stays at the brace. */
#include <stdlib.h>
int main(void)
{
  int *p = malloc(sizeof(int));
  if (!p)
    return 2;
  if (*p)
    return 1;
  return 0;
}
