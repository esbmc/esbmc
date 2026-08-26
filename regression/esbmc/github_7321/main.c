#include <assert.h>

struct P
{
  double x;
};

union U
{
  double d;
};

int main(void)
{
  int c = 0;

  /* if(c, x, y) -> x where the arms are IEEE-equal but differ in sign. */
  double d = c ? -0.0 : 0.0;
  assert(__builtin_signbit(d) == 0);

  struct P sa = {-0.0}, sb = {0.0};
  struct P sr = c ? sa : sb;
  assert(__builtin_signbit(sr.x) == 0);

  _Complex double z = c ? -0.0 - 0.0i : 0.0 + 0.0i;
  assert(__builtin_signbit(__real__ z) == 0);

  /* with(s, f, v) -> s where v is IEEE-equal to the stored element. */
  double arr[4] = {0.0, 0.0, 0.0, 0.0};
  arr[2] = -0.0;
  assert(__builtin_signbit(arr[2]) == 1);

  double neg[4] = {-0.0, -0.0, -0.0, -0.0};
  neg[1] = 0.0;
  assert(__builtin_signbit(neg[1]) == 0);

  struct P p = {0.0};
  p.x = -0.0;
  assert(__builtin_signbit(p.x) == 1);

  union U u = {0.0};
  u.d = -0.0;
  assert(__builtin_signbit(u.d) == 1);

  return 0;
}
