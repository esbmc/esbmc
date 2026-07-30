int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  // The same two properties with the violable one last, which is detected:
  // pins the ordering dependence in #6540 rather than the flag pair alone.
  __ESBMC_assert(x >= 0 || x < 0, "holds");
  __ESBMC_assert(x != 42, "may fail");
  return 0;
}
