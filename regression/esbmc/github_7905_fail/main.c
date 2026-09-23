// #7905: a cast to lanes of another width reads the bits back, not the values.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));
typedef short v8s __attribute__((__vector_size__(16)));

int main(void)
{
  v4i s = {0x00020001, 0, 0, 0};
  v8s h = (v8s)s;
  assert(h[1] != 2);
  return 0;
}
