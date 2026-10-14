// #7907: the write through the pointer reaches the vector it points at.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));

int main(void)
{
  v4i s = {1, 2, 3, 4};
  v4i *p = &s;
  *p = (v4i){5, 6, 7, 8};
  assert(s[0] == 1);
  return 0;
}
