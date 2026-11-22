/* #10081: without --force-malloc-success, calloc may still fail under --sv-comp. */
#include <stdlib.h>
extern void reach_error(void);
int main()
{
  int *p = calloc(4, sizeof(int));
  if (p == 0)
    reach_error();
  return 0;
}
