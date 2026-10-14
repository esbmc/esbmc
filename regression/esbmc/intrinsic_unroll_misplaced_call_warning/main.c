/* Test that __ESBMC_unroll(N) is reported as misplaced when an unrelated
 * statement (here a function call) sits between the intrinsic and the loop:
 * the intrinsic must directly precede the loop. The loop is still bounded
 * by --unwind so verification succeeds, but the intrinsic is not applied.
 */

void g();

int main()
{
  int sum = 0;

  __ESBMC_unroll(2);
  g();
  for(int i = 0, j = 10; i < j; i++, j--)
    sum += i;

  return 0;
}
