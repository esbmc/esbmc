/* cpp_assigns_pass:
 * --enforce-contract with assigns clause on a C++ free function.
 * foo only modifies 'x' which is listed in __ESBMC_assigns, so
 * assigns compliance must pass.
 *
 * Expected: VERIFICATION SUCCESSFUL
 */

int x = 0;
int y = 0;

void foo(void)
{
  __ESBMC_assigns(x);
  __ESBMC_ensures(x == 1);
  x = 1;
  /* y is not modified — compliance passes */
}
