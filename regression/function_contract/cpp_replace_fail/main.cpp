/* cpp_replace_fail:
 * --replace-call-with-contract on a C++ free function.
 * The contract says counter == old(counter) + 5 but the caller asserts
 * counter == 1, which contradicts the assumed ensures clause.
 *
 * Regression for: matches_replace_pattern not stripping the trailing '#'
 * from C++ USR symbol names and is_compiler_generated falsely treating
 * C++ free functions as compiler-generated.  Both bugs prevented the call
 * from being replaced, so the inline __ESBMC_ensures ASSUME silenced the
 * assert and the result was wrongly VERIFICATION SUCCESSFUL.
 *
 * Expected: VERIFICATION FAILED
 */

int counter = 0;

void increment(void)
{
  __ESBMC_requires(counter >= 0);
  __ESBMC_ensures(counter == __ESBMC_old(counter) + 5);
  counter++;
}

int main()
{
  increment();
  __ESBMC_assert(counter == 1, "counter should be 1 (body ran)");
  return 0;
}
