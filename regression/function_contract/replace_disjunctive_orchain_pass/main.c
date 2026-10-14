/* replace_disjunctive_orchain_pass (issue #6298):
 * A pure ||-chain ensures (three disjuncts, no nested &&) each referencing
 * old(): the flattening must keep every disjunct rather than collapsing the
 * chain. f leaves a unchanged, so a == old(a) (the first disjunct) holds.
 * Expected: VERIFICATION SUCCESSFUL
 */
#include <stddef.h>

typedef struct
{
  int a;
} S;

void f(S *s)
{
  __ESBMC_requires(s != NULL);
  __ESBMC_assigns(s->a);
  __ESBMC_ensures(
    s->a == __ESBMC_old(s->a) || s->a == __ESBMC_old(s->a) + 1 ||
    s->a == __ESBMC_old(s->a) + 2);
}

int main(void)
{
  S s;
  s.a = 5;
  f(&s);
  __ESBMC_assert(
    s.a == 5 || s.a == 6 || s.a == 7, "a stays within old..old+2");
  return 0;
}
