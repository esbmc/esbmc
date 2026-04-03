/* Test: __ESBMC_old() in loop invariant correctly captures pre-loop value.
 *
 * Previously, replace_old_with_snapshots() could not match the
 * return_value$___ESBMC_old_raw$N symbol to any active snapshot, so
 * __ESBMC_old(y) evaluated to the POST-havoc (nondet) value of y instead
 * of the pre-loop snapshot.
 *
 * The fix (patch_old_snapshot_assigns) replaces the old_snapshot side effect
 * assignment with &snapshot_of_y before emitting the ASSUME, so
 * *(T*)return_value$... correctly dereferences the snapshot.
 */
int main()
{
  int x = 0;
  int y = 10;

  /* Loop only modifies x. Invariant: y retains its pre-loop value. */
  __ESBMC_loop_assigns(x);
  __ESBMC_loop_invariant(x >= 0 && y == __ESBMC_old(y));
  while (x < 5)
    x++;

  __ESBMC_assert(y == 10, "y unchanged by loop");
  return 0;
}
