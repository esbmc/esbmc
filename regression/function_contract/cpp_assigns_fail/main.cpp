/* cpp_assigns_fail:
 * --enforce-contract with assigns clause on a C++ free function.
 * foo modifies 'y' which is NOT listed in __ESBMC_assigns(x), so
 * assigns compliance must report a violation.
 *
 * Expected: VERIFICATION FAILED
 */

int x = 0;
int y = 0;

void foo(void)
{
  __ESBMC_assigns(x);
  __ESBMC_ensures(x == 1);
  x = 1;
  y = 2; /* y is not in assigns — compliance fails */
}
