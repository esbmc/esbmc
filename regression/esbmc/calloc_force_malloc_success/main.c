#include <assert.h>
#include <stdlib.h>
int main()
{
  int *p = calloc(1, sizeof(int));
  assert(p != NULL);
  free(p);
  return 0;
}
