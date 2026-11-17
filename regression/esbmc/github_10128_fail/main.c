/* A real violation is still found under --bidirectional, and is reported
   against the program's assertion (esbmc/esbmc#10128). */
int main()
{
  unsigned x = 0;
  do
  {
    x++;
    __ESBMC_assert(x != 5, "five");
  } while (x < 5);
}
