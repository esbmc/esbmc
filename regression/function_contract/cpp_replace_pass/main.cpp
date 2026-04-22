/* cpp_replace_pass:
 * --replace-call-with-contract on a C++ free function.
 * The contract correctly describes the body (counter += 1), so the caller's
 * assertion counter == 1 must hold after replacement.
 *
 * Regression for: matches_replace_pattern not stripping the trailing '#'
 * from C++ USR symbol names, causing zero replacements, and
 * is_compiler_generated falsely skipping the call site due to '#' in name.
 *
 * Expected: VERIFICATION SUCCESSFUL
 */

int counter = 0;

void increment(void)
{
  __ESBMC_requires(counter >= 0);
  __ESBMC_ensures(counter == __ESBMC_old(counter) + 1);
  counter++;
}

int main()
{
  increment();
  __ESBMC_assert(counter == 1, "counter should be 1");
  return 0;
}
