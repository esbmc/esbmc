// q is what __ESBMC_memset returns, so the loop's write through q reaches c
// and the inductive step must havoc c. Were q taken to point nowhere, c.n
// would start the step at 0 and the step would prove the property; the base
// case reaches the violation at k = 6.
#include <assert.h>
#include <string.h>

struct ctx
{
  int *p;
  int n;
};

int nondet_int(void);

int main(void)
{
  struct ctx c;
  struct ctx *q = memset(&c, 0, sizeof(c));
  while (nondet_int())
  {
    if (c.n < 10)
      q->n++;
    assert(c.n <= 5);
  }
  return 0;
}
