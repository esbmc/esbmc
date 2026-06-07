/* Regression for issue #5187 (soundness guard):
 * main's implicit return is 0 (C11 §5.1.2.2.3p1), so a postcondition claiming a
 * different value must be reported as violated. Before the fix the per-main
 * END_FUNCTION special-casing assumed false on the renamed original's exit,
 * killing the path before the wrapper's ensures assertion — so this false
 * postcondition passed vacuously (0 VCCs).
 */
__ESBMC_contract
int main(void)
{
  __ESBMC_ensures(__ESBMC_return_value == 42);
}
