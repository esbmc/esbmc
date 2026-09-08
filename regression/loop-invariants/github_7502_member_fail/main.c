/* Regression: GitHub #7502 -- a struct member reached through a pointer.  The
 * loop drives `sp->c` to 0, so the exit edge only exists once the pointee is
 * havoc'd; before, it was infeasible and this assertion was never reached. */
#include <assert.h>

struct S
{
  int c;
};

static void run(struct S *sp)
{
  __ESBMC_loop_invariant(sp->c >= 0);
  while (sp->c > 0)
  {
    sp->c--;
  }
  assert(0);
}

int main()
{
  struct S s;
  s.c = 4;
  run(&s);
  return 0;
}
