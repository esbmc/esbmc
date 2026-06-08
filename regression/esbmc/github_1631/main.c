#include <stdlib.h>
#include <assert.h>

int main()
{
  void *a = (void *)malloc(4);
  void *b = (void *)malloc(-4);
  assert(0);
}
