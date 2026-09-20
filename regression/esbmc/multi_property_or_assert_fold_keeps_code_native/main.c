/* As multi_property_or_assert_fold_keeps_code, via the native converter
   (esbmc/esbmc#7900). */
int main()
{
  int c = nondet_int();
  int x = 0;
  if (c)
  {
    __ESBMC_assert(0, "v");
    x = 1;
  }
  __ESBMC_assert(x == 0, "w");
  return 0;
}
