/* replace_disjunctive_4term_pass (issue #6298):
 * A disjunct with a FOUR-term conjunct guarding four old() fields, to exercise
 * deeper nesting of the short-circuit temporaries than the 3-term case.
 * Expected: VERIFICATION SUCCESSFUL
 */
#include <stddef.h>

typedef struct
{
  int a;
  int b;
  int c;
  int d;
  int trigger;
} S;

void f(S *s)
{
  __ESBMC_requires(s != NULL);
  __ESBMC_assigns(s->a, s->b, s->c, s->d);
  __ESBMC_ensures(
    (s->trigger == 1 && s->a == 99 && s->b == 99 && s->c == 99 &&
     s->d == 99) ||
    (s->trigger != 1 && s->a == __ESBMC_old(s->a) &&
     s->b == __ESBMC_old(s->b) && s->c == __ESBMC_old(s->c) &&
     s->d == __ESBMC_old(s->d)));
  if (s->trigger == 1)
  {
    s->a = 99;
    s->b = 99;
    s->c = 99;
    s->d = 99;
  }
}

int main(void)
{
  S s;
  s.trigger = 0;
  s.a = 1;
  s.b = 2;
  s.c = 3;
  s.d = 4;
  f(&s);
  __ESBMC_assert(
    s.a == 1 && s.b == 2 && s.c == 3 && s.d == 4, "all four fields unchanged");
  return 0;
}
