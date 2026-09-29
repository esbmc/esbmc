/* --bidirectional renumbers the program mid-run; one assertion stays one row
   (esbmc/esbmc#7900). */
int main()
{
  unsigned x = 0;
  while (x < 10)
  {
    x += 2;
    __ESBMC_assert(x != 11, "inloop");
  }
  __ESBMC_assert(x == 10, "after");
  return 0;
}
