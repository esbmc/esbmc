/* replace_disjunctive_multiterm_and_pass (issue #6298):
 * --replace-call-with-contract with a disjunctive ensures whose untaken
 * branch is a 3-term conjunct guarding two old() fields:
 *   (trigger==1 && a==99 && b==99) ||
 *   (trigger!=1 && a==old(a) && b==old(b))
 *
 * Clang lowers the 3-term chain into nested short-circuit temporaries; the
 * leading conjuncts (trigger!=1, a==old(a)) live in the guard of a nested
 * temporary, not in an assignment to the outer one. The reconstruction used
 * to keep only the innermost assignment (b==old(b)) and drop the rest, leaving
 * `a` unconstrained so this correct program was spuriously rejected.
 *
 * Expected: VERIFICATION SUCCESSFUL
 */
#include <stddef.h>

typedef struct
{
  int a;
  int b;
  int trigger;
} S;

void f(S *s)
{
  __ESBMC_requires(s != NULL);
  __ESBMC_assigns(s->a, s->b);
  __ESBMC_ensures(
    (s->trigger == 1 && s->a == 99 && s->b == 99) ||
    (s->trigger != 1 && s->a == __ESBMC_old(s->a) &&
     s->b == __ESBMC_old(s->b)));
  if (s->trigger == 1)
  {
    s->a = 99;
    s->b = 99;
  }
}

int main(void)
{
  S s;
  s.trigger = 0;
  s.a = 5;
  s.b = 7;
  f(&s);
  __ESBMC_assert(
    s.a == 5 && s.b == 7, "both fields unchanged when trigger != 1");
  return 0;
}
