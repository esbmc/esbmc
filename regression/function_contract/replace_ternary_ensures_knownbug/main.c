/* replace_ternary_ensures_knownbug (issue #6298):
 * The recursive flattening only descends &&/|| trees, not a ternary (?:)
 * ensures. A ternary is lowered soundly (the unsound-if-passed t==1 case is
 * correctly rejected) but imprecisely: the else-branch old() constraints are
 * not recovered, so this correct program is spuriously rejected. The desired
 * result is VERIFICATION SUCCESSFUL; until ternary ensures are flattened it is
 * absent (VERIFICATION FAILED), so this is pinned KNOWNBUG.
 */
#include <stddef.h>

typedef struct
{
  int a;
  int b;
  int t;
} S;

void f(S *s)
{
  __ESBMC_requires(s != NULL);
  __ESBMC_assigns(s->a, s->b);
  __ESBMC_ensures(
    s->t == 1 ? (s->a == 99 && s->b == 99)
              : (s->a == __ESBMC_old(s->a) && s->b == __ESBMC_old(s->b)));
  if (s->t == 1)
  {
    s->a = 99;
    s->b = 99;
  }
}

int main(void)
{
  S s;
  s.t = 0;
  s.a = 5;
  s.b = 7;
  f(&s);
  __ESBMC_assert(s.a == 5 && s.b == 7, "else branch: both fields unchanged");
  return 0;
}
