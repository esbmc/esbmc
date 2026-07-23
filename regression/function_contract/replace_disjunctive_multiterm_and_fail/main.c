/* replace_disjunctive_multiterm_and_fail (issue #6298):
 * Negative counterpart of replace_disjunctive_multiterm_and_pass. The ensures
 * pins a to its old value (5) when trigger != 1, so asserting a == 99 must
 * fail. This guards against the fix over-correcting into a vacuously-true
 * ASSUME that would accept any post-state.
 *
 * Expected: VERIFICATION FAILED
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
  __ESBMC_assert(s.a == 99, "a must NOT be 99 when trigger != 1");
  return 0;
}
