/* replace_ternary_ensures (issue #6298, fixed by #6499):
 * A ternary ensures used to be reconstructed as its else-arm alone, so the
 * then-arm and the guard were lost and this correct program was spuriously
 * rejected. Reconstructing the conditional recovers both arms.
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
