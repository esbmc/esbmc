int main()
{
  int a, b, c, d, e;
  //------------------------------------------------------------
  __ESBMC_assert( a * b == b * a, "assertion 1");
  __ESBMC_assert( a * c == c * a, "assertion 2");
}
