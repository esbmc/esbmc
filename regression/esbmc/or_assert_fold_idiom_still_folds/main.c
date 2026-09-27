/* The idiom's trailing 0 is a no-op, so it still folds (esbmc/esbmc#7900). */
int main()
{
  int c = nondet_int();
  (void)((c) || (__ESBMC_assert(0, "v"), 0));
  return 0;
}
