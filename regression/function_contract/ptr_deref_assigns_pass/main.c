/* ptr_deref_assigns_pass:
 * NOTE: --assume-nonnull-valid required for Phase 2C: ESBMC must allocate a
 * real object for pointer params so that writes through *p are tracked in the
 * symbolic state. Without it, nondet pointers have no associated object and
 * the write *p=val is invisible to the post-call assertion.
 * Function declares __ESBMC_assigns(*p) — allowed to modify *p.
 * Body correctly increments *p.
 *
 * Phase 2C: pointer-parameter dereference assigns compliance.
 * Expected: VERIFICATION SUCCESSFUL
 */
int increment(int *p)
{
  __ESBMC_requires(p != (int *)0);
  __ESBMC_assigns(*p);
  __ESBMC_ensures(__ESBMC_return_value == 0);
  *p = *p + 1;
  return 0;
}

int main() { return 0; }
