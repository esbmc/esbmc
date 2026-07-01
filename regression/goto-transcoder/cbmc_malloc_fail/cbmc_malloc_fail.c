#include <stdlib.h>
#include <assert.h>
int main()
{
  int *p = malloc(sizeof(int));
  assert(p != 0);
  return 0;
}
