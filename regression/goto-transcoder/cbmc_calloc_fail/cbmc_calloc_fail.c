#include <stdlib.h>
#include <assert.h>
int main()
{
  int *p = calloc(1, sizeof(int));
  if (p)
    assert(*p == 1);
  return 0;
}
