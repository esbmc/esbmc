/* is_fresh_loopinv_poweroftwo_fail (issue #6399):
 * Negative control for is_fresh_loopinv_poweroftwo_pass. The loop only touches
 * vec[0..2], so vec[3] is left havoc'd (nondet) and the ensures over all four
 * elements must FAIL. Confirms the #6399 fix (a wider array index domain) does
 * not make the power-of-two case vacuously verify.
 */
typedef struct { int x; } Elem;
typedef struct { Elem vec[4]; } Vec;

void touch(Elem *e)
{
  __ESBMC_requires(__ESBMC_is_fresh(e, sizeof(Elem)));
  __ESBMC_ensures(e->x == 0);
  __ESBMC_assigns(e->x);
  e->x = 0;
}

void touch_all(Vec *v)
{
  unsigned k, j;
  __ESBMC_requires(__ESBMC_is_fresh(v, sizeof(Vec)));
  __ESBMC_assigns(v->vec);
  __ESBMC_ensures(__ESBMC_forall(&j, !(j < 4) || v->vec[j].x == 0));

  __ESBMC_loop_invariant(k <= 3 && __ESBMC_forall(&j, !(j < k) || v->vec[j].x == 0));
  for (k = 0; k < 3; k++)
    touch(&v->vec[k]);
}

int main(void) { return 0; }
