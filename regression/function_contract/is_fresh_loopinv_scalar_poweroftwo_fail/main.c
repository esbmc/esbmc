/* is_fresh_loopinv_scalar_poweroftwo_fail (issue #6399):
 * Negative control for is_fresh_loopinv_scalar_poweroftwo_pass. The loop only
 * touches v[0..2], so v[3] stays havoc'd and the ensures over all four
 * elements must FAIL -- the widened index domain must not make the
 * power-of-two case vacuously verify.
 */
void touch(int *e)
{
  __ESBMC_requires(__ESBMC_is_fresh(e, sizeof(int)));
  __ESBMC_ensures(*e == 0);
  __ESBMC_assigns(*e);
  *e = 0;
}

void touch_all(int *v)
{
  unsigned k, j;
  __ESBMC_requires(__ESBMC_is_fresh(v, 4 * sizeof(int)));
  __ESBMC_assigns(*v);
  __ESBMC_ensures(__ESBMC_forall(&j, !(j < 4) || v[j] == 0));

  __ESBMC_loop_invariant(k <= 3 && __ESBMC_forall(&j, !(j < k) || v[j] == 0));
  for (k = 0; k < 3; k++)
    touch(&v[k]);
}

int main(void) { return 0; }
