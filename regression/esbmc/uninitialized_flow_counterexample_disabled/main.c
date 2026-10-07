int main(void)
{
  unsigned initialized;
  unsigned counter = initialized;
  counter++;
  __ESBMC_assert(0, "expected failure");
  return 0;
}
