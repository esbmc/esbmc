#include <stdlib.h>
#include <assert.h>
int main()
{
  int *p = malloc(sizeof(int));
  if (p)
  {
    *p = 5;
    assert(*p == 5);
  }
  return 0;
}
