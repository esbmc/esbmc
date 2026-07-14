#include <stdlib.h>
#include <assert.h>

// Regression for esbmc/esbmc#5337: under --no-simplify the sizeof node is not
// folded by the simplifier, so the SMT backend must lower it to its
// eagerly-computed byte-size value rather than aborting with
// "Couldn't convert expression in unrecognised format".

struct T
{
  int a;
  long b;
};

int main(void)
{
  unsigned long s = sizeof(struct T);
  assert(s == sizeof(int) + sizeof(long) || s > sizeof(int));

  struct T *p = malloc(sizeof(struct T));
  if (p)
  {
    p->a = 3;
    assert(p->a == 3);
    free(p);
  }
  return 0;
}
