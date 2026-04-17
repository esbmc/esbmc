/* ptr_deref_assigns_fail:
 * NOTE: --assume-nonnull-valid required for Phase 2C: ESBMC must allocate a
 * real object for pointer params so that writes through *p are tracked in the
 * symbolic state. Without it, nondet pointers have no associated object and
 * the write *p=val is invisible to the post-call assertion.
 * Function declares __ESBMC_assigns(g) but also writes to *p (undeclared).
 * This violates the assigns clause.
 *
 * Phase 2C: pointer-parameter dereference assigns compliance.
 * Expected: VERIFICATION FAILED
 */
int g = 0;

int update(int *p, int val)
{
  __ESBMC_requires(p != (int *)0);
  __ESBMC_assigns(g);           /* only g is declared */
  __ESBMC_ensures(__ESBMC_return_value == 0);
  g = val;
  *p = val; /* BUG: *p not in assigns clause */
  return 0;
}

int main() { return 0; }
