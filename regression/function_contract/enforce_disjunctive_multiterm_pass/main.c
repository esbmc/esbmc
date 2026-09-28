/* enforce_disjunctive_multiterm_pass (issue #6298):
 * The --enforce-contract direction (extract_ensures_from_body) must also keep
 * every conjunct of a disjunctive multi-term ensures. f's body satisfies the
 * contract, so enforcing it verifies.
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
