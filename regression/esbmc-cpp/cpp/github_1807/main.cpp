#include <stdlib.h>
#include <assert.h>

int main()
{
  void *p = realloc(NULL, 0);
  assert(p);
}
