// Widening the decay must not widen an element the source really did name.
// `__ESBMC_assigns(a[0])` still frames one element, so the caller keeps the
// rest, which is the whole point of a precise assigns clause.
#include <assert.h>

int a[4];

void f(void)
{
  __ESBMC_assigns(a[0]);
  __ESBMC_ensures(a[0] == 1);
  a[0] = 1;
}

int main(void)
{
  a[1] = 20;
  f();
  assert(a[0] == 1);
  assert(a[1] == 20); /* outside the frame, so it survives the call */
  return 0;
}
