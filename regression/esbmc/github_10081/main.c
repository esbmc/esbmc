/* #10081: --force-malloc-success makes calloc succeed under --sv-comp too. */
#include <stdlib.h>
extern void reach_error(void);
int main()
{
  int *p = calloc(4, sizeof(int));
  if (p == 0)
    reach_error();
  return 0;
}
