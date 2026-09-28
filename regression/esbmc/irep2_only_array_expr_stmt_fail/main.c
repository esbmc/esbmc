/* The discarded statement must stay discarded: rewriting it may not disturb the
   member it names. */
#include <assert.h>

struct Base
{
  int ss[128];
};

int main(void)
{
  struct Base x, *y = &x;

  x.ss[0] = 5;
  y->ss;

  assert(x.ss[0] == 6);
  return 0;
}
