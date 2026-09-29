// #7905: under --ir there are no bits to lay out as lanes of another width, so
// such a vector bitcast is refused rather than encoded.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));
typedef short v8s __attribute__((__vector_size__(16)));

int main(void)
{
  v4i s = {0x00020001, 0, 0, 0};
  v8s h = (v8s)s;
  assert(h[0] == 1);
  return 0;
}
