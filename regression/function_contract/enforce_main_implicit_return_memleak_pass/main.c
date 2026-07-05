/* Regression for issue #5187:
 * C11 §5.1.2.2.3p1 mandates that reaching the } that terminates main returns 0.
 * A contract ensures(__ESBMC_return_value == 0) on a main with no explicit
 * return must therefore hold, even when --memory-leak-check is enabled (the
 * leak-check path used to expose a nondet implicit return value).
 */
__ESBMC_contract
int main(void)
{
  __ESBMC_ensures(__ESBMC_return_value == 0);
}
