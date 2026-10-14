// #7907: the lanes must survive the load through the pointer in order.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));

int main(void)
{
  v4i s = {1, 2, 3, 4};
  v4i *p = &s;
  v4i c = *p;
  assert(c[0] == 4);
  return 0;
}
