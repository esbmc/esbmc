/* Regression: GitHub #7478 -- an array element reached through a pointer.
 * `modifies_pointer_array` reports this shape but only goto_k_induction reads
 * it, so the invariant schema has to recognise it too. */
#include <assert.h>

struct S
{
  int e[4];
};

static void run(struct S *s)
{
  __ESBMC_loop_invariant(s->e[0] >= 0);
  while (s->e[0] > 0)
  {
    s->e[0]--;
  }
  assert(0);
}

int main()
{
  struct S s;
  s.e[0] = 4;
  run(&s);
  return 0;
}
