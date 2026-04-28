/* cpp_enforce_pass:
 * --enforce-contract on a C++ free function.
 * The ensures says counter == old(counter) + 1 and the body does exactly
 * that, so contract enforcement must succeed.
 *
 * Regression for: find_function_symbol not trying the C++ USR suffix '#',
 * which caused "Function X not found" and vacuous VERIFICATION SUCCESSFUL.
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
