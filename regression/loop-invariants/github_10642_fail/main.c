/* Regression: GitHub #10642, positive control. Every state the invariant
 * admits at loop exit has i == 10, so reach_error is forced and stays FAILED. */
extern void reach_error(void);

int main()
{
  int i = 0;
  __ESBMC_loop_invariant(0 <= i && i <= 10);
  while (i < 10)
    i++;
  if (i == 10)
    reach_error();
  return 0;
}
