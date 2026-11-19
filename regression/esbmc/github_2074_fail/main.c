/* #2074: the nil argument must leave x unconstrained, not fix its interval. */
int f(unsigned char x)
{
  __ESBMC_assert(x != 200, "x can be 200");
  return x;
}
