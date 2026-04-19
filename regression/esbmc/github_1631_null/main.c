#include <stdlib.h>
#include <assert.h>

int main()
{
  void *b = (void *)malloc(-4);
  assert(b == NULL);
}
