/* An invariant that calls a function, on a loop preceded by another loop. The
 * pass moves the call's temporaries out of the body; erasing them dangled the
 * first loop's exit target and compute_target_numbers aborted. */
int g(int k)
{
  return k;
}

int main(void)
{
  int i, total = 0;

  for (i = 0; i < 2; i++)
    total += i;

  __ESBMC_loop_invariant(total == g(i));
  for (i = 0; i < 2; i++)
    total += i;

  __ESBMC_assert(total >= 0, "reached a verdict rather than aborting");
  return 0;
}
