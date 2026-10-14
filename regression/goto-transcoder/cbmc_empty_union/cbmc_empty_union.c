#include <assert.h>

/* Reading the struct out of the pun makes the SMT backend ask for the
   zero-width union member's zero value. */
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
  assert(p.s.x == 0);
  return 0;
}
