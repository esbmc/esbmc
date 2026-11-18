/* #2074: --function leaves the entry point's arguments nil, and interval
 * analysis dereferenced them. */
int f(unsigned char x)
{
  __ESBMC_assert(x <= 255, "x fits in a byte");
  return x;
}
