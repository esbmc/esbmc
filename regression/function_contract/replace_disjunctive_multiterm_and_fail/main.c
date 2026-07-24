/* replace_disjunctive_multiterm_and_fail (issue #6298):
 * Negative counterpart of replace_disjunctive_multiterm_and_pass, exercising
 * the *other* (trigger==1) disjunct: the ensures pins both fields to 99, so
 * asserting b == 7 after the call must fail. Guards that the taken disjunct's
 * constraints are actually applied (not dropped or vacuously assumed).
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
  s.trigger = 1;
  s.a = 5;
  s.b = 7;
  f(&s);
  __ESBMC_assert(s.b == 7, "b must be 99 when trigger == 1");
  return 0;
}
