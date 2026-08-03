/* github_6483_struct_extent_knownbug:
 *   A struct pointer parameter keeps a single-element stack backing, so s->x
 *   is admitted even though the contract states no extent for s. Accesses
 *   past that one element are still caught, so only the first element is
 *   unjustified.
 *
 *   Moving struct params onto the nondet-extent heap allocation used for
 *   other pointer types would close this, but a heap-backed struct silently
 *   discharges __ESBMC_old-based ensures clauses (#6483), which is a worse
 *   false negative. Flip this test to expect VERIFICATION FAILED once #6483
 *   is fixed and the struct branch moves to emit_pointer_param_malloc.
 *
 *   Expected (correct): VERIFICATION FAILED
 *   Current (bug):      VERIFICATION SUCCESSFUL - unjustified first element
 */
typedef struct
{
  int x;
} S;

void f(S *s)
{
  __ESBMC_requires(s != 0);
  __ESBMC_ensures(1);
  s->x = 1;
}

int main()
{
  return 0;
}
