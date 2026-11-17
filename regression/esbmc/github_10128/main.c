/* --bidirectional must not report its own blocking assertion as a bug
   (esbmc/esbmc#10128). */
int main()
{
  unsigned x = 0;
  do
  {
    x++;
    __ESBMC_assert(x != 6, "six");
  } while (x < 5);
}
