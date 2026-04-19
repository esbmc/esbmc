/* Test __ESBMC_unroll(N) applied in multiple functions.
 * main() uses __ESBMC_unroll(5) before a for loop (bound 5).
 * foo() uses __ESBMC_unroll(10) before a while loop (bound 10).
 * Both loops should be independently annotated and verification succeeds.
 */

int foo();

int main()
{
  __ESBMC_unroll(5);
  for(int i = 0; i < 5; i++)
    ;

  foo();
  return 0;
}

int foo()
{
  int i = 0;
  __ESBMC_unroll(10);
  while(i < 10)
    i++;
  return i;
}
