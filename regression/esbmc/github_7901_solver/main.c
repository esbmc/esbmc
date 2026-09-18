/* esbmc/esbmc#7901: naming a solver ESBMC was not built with is a rejected
 * input, not a crash. The name below is in no build, so the test does not
 * depend on which backends were compiled in. */
int main(void)
{
  return 0;
}
