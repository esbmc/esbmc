#include <assert.h>

struct P
{
  double x;
};

int main(void)
{
  int c = 0;

  double d = c ? -0.0 : 0.0;
  assert(__builtin_signbit(d) == 0);

  struct P a = {-0.0}, b = {0.0};
  struct P r = c ? a : b;
  assert(__builtin_signbit(r.x) == 0);

  _Complex double z = c ? -0.0 - 0.0i : 0.0 + 0.0i;
  assert(__builtin_signbit(__real__ z) == 0);

  return 0;
}
