/* cpp_enforce_fail:
 * --enforce-contract on a C++ free function.
 * The ensures says counter == old(counter) + 2 but the body only does +1,
 * so contract enforcement must report a violation.
 *
 * Regression for: find_function_symbol not trying the C++ USR suffix '#',
 * which caused "Function X not found" and vacuous VERIFICATION SUCCESSFUL
 * even when the ensures clause was clearly wrong.
 *
 * Expected: VERIFICATION FAILED
 */

int counter = 0;

void increment(void)
{
  __ESBMC_requires(counter >= 0);
  __ESBMC_ensures(counter == __ESBMC_old(counter) + 2);
  counter++;
}
