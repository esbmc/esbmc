#include <stdlib.h>
#include <assert.h>
int main()
{
  int *p = calloc(2, sizeof(int));
  assert(p[0] == 0 && p[1] == 0); // no NULL guard: --force-malloc-success
  return 0;
}
