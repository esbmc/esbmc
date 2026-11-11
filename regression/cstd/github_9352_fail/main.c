// #9352: (_Bool)(1 + 1) is 1, so the post-increment leaves s.b set.
#include <assert.h>

struct S
{
  _Bool b : 1;
} s;

int main(void)
{
  s.b = 1;
  int old = s.b++;
  assert(old == 1);
  assert(s.b == 0);
  return 0;
}
