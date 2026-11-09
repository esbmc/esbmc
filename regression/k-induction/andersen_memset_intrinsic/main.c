// memset's model calls __ESBMC_memset, which only stores the fill byte. The
// points-to analysis used to widen it to "writes anything", so the pointer
// later loaded from c pointed anywhere and the loop's write through it
// disabled the inductive step.
#include <assert.h>
#include <string.h>

struct ctx
{
  int *p;
  int n;
};

int buf[4];
int nondet_int(void);

int main(void)
{
  struct ctx c;
  memset(&c, 0, sizeof(c));
  c.p = buf;
  int x = nondet_int();
  __ESBMC_assume(x > 0);
  while (nondet_int())
  {
    c.p[0] = 1;
    assert(x > 0);
  }
  return 0;
}
