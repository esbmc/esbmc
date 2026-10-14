// #7905: on a big-endian target a lane's bytes are the other way round, which
// shows when a cast changes the lane width or goes to or from a scalar.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));
typedef short v8s __attribute__((__vector_size__(16)));
typedef unsigned char v16c __attribute__((__vector_size__(16)));
typedef int v2i __attribute__((__vector_size__(8)));

int main(void)
{
  v4i s = {0x00020001, 0x04030201, 0, -1};
  v8s h = (v8s)s;
  assert(h[0] == 2 && h[1] == 1 && h[7] == -1);
  v16c c = (v16c)s;
  assert(c[4] == 4 && c[7] == 1);

  v2i v = {1, 2};
  long long x = (long long)v;
  assert(x == 0x0000000100000002LL);
  v2i w = (v2i)x;
  assert(w[0] == 1 && w[1] == 2);
  return 0;
}
