#include <assert.h>

/* The negative twin of cbmc_empty_union: the assertion must be reported,
   not lost. */
union empty
{
};

struct s
{
  union empty e;
  int x;
};

union pun
{
  struct s s;
  long l;
};

int main()
{
  union pun p;
  __CPROVER_assume(p.l == 0);
  assert(p.s.x == 1);
  return 0;
}
