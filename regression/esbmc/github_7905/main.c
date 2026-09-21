// #7905: a cast between a vector and another type of its size reinterprets
// the bits.
#include <assert.h>

typedef int v4i __attribute__((__vector_size__(16)));
typedef unsigned v4u __attribute__((__vector_size__(16)));
typedef short v8s __attribute__((__vector_size__(16)));
typedef float v4f __attribute__((__vector_size__(16)));
typedef int v2i __attribute__((__vector_size__(8)));

int nondet_int(void);

int main(void)
{
  v4i c = {-1, 0, 0, 0};
  v4u d = (v4u)c;
  assert(d[0] == 0xffffffffu);
  v4u e = c;
  assert(e[0] == 0xffffffffu);

  v4i b = (v4i)(v4f){1, 2, 3, 4};
  assert(b[0] == 0x3f800000 && b[3] == 0x40800000);

  v4i s = {0x00020001, 0x00040003, 0, -1};
  v8s h = (v8s)s;
  assert(h[0] == 1 && h[1] == 2 && h[3] == 4 && h[7] == -1);

  v2i v = {1, 2};
  long long x = (long long)v;
  assert(x == 0x0000000200000001LL);
  v2i w = (v2i)x;
  assert(w[1] == 2);

  v4i a = {nondet_int(), 2, 3, 4};
  v4u m = (v4u)(a == (v4i){1, 2, 0, 4});
  assert(m[1] == 0xffffffffu && m[2] == 0);
  assert(m[0] == 0 || m[0] == 0xffffffffu);
  return 0;
}
