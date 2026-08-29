/* is_fresh_loopinv_scalar_poweroftwo_pass (issue #6399):
 * The scalar-pointee counterpart of is_fresh_loopinv_poweroftwo_pass. The
 * fresh object is a plain int buffer, so nothing here is a struct: 16 bytes
 * (2^4) used to report a false VERIFICATION FAILED while 20 bytes verified,
 * the same power-of-two parity coming from the array index domain being one
 * bit too narrow to hold the one-past-the-end index. Kept as a separate test
 * because a contract-side fix that retypes aggregate is_fresh objects cannot
 * reach this shape -- only widening the domain does.
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

  __ESBMC_loop_invariant(k <= 4 && __ESBMC_forall(&j, !(j < k) || v[j] == 0));
  for (k = 0; k < 4; k++)
    touch(&v[k]);
}

int main(void) { return 0; }
