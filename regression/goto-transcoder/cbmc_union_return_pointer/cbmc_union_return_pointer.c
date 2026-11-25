#include <assert.h>
const unsigned lo = 7, hi = 7;
_Bool nondet_bool(void);
struct incl { const unsigned *p; };
struct bound { union { struct incl included; struct incl excluded; } cases; };
struct bound get(const unsigned *self)
{
  struct bound b;
  const unsigned *start = self;
  b.cases.included = (struct incl){start};
  return b;
}
int main(void)
{
  struct bound r = get(nondet_bool() ? &lo : &hi);
  const unsigned *s = r.cases.included.p; assert(*s == 7);
  return 0;
}
