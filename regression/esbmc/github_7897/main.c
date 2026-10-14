// #7897: a vector comparison yields a lane mask, not a scalar.
#include <assert.h>

typedef float v4f __attribute__((__vector_size__(16)));
typedef int v4i __attribute__((__vector_size__(16)));
typedef double v2d __attribute__((__vector_size__(16)));
typedef long long v2l __attribute__((__vector_size__(16)));
typedef unsigned char v16u8 __attribute__((__vector_size__(16)));
typedef signed char v16s8 __attribute__((__vector_size__(16)));
typedef _Bool bool4 __attribute__((ext_vector_type(4)));

static int calls;

static v4f next(void)
{
  ++calls;
  return (v4f){1, 2, 3, 4};
}

int main(void)
{
  v4f a = {1, 2, 3, 4}, b = {1, 0, 3, 0};
  v4i m = a == b;
  assert(m[0] == -1 && m[1] == 0 && m[2] == -1 && m[3] == 0);

  v4i ne = a != b, lt = a < b, gt = a > b, le = a <= b, ge = a >= b;
  assert(ne[0] == 0 && ne[1] == -1);
  assert(lt[0] == 0 && lt[1] == 0);
  assert(gt[0] == 0 && gt[1] == -1);
  assert(le[0] == -1 && le[1] == 0);
  assert(ge[0] == -1 && ge[1] == -1);

  v4i bits = __builtin_bit_cast(v4i, a == b);
  assert(bits[2] == -1 && bits[3] == 0);

  v4i splat = a == 3;
  assert(splat[2] == -1 && splat[0] == 0);

  v4f nan = {__builtin_nanf(""), 0, 0, 0};
  v4i eq_nan = nan == nan, ne_nan = nan != nan;
  assert(eq_nan[0] == 0 && ne_nan[0] == -1);

  v2d c = {1, 2}, d = {1, 3};
  v2l cd = c < d;
  assert(cd[0] == 0 && cd[1] == -1);

  v16u8 x = {200, 1}, y = {1, 200};
  v16s8 xy = x > y;
  assert(xy[0] == -1 && xy[1] == 0 && xy[15] == 0);

  bool4 p = {1, 0, 1, 0}, q = {1, 1, 0, 0};
  bool4 pq = p == q;
  assert(pq[0] && !pq[1]);

  v4i once = next() == a;
  assert(calls == 1 && once[3] == -1);

  int guard = 0;
  if (guard && (next() == a)[0])
    return 1;
  assert(calls == 1);

  int hits = 0;
  for (int i = 0; i < 3; i++)
    if ((next() == a)[0])
      hits++;
  assert(calls == 4 && hits == 3);
  return 0;
}
