#if defined(__unix__) || defined(__APPLE__) || defined(__MACH__)
#  include <strings.h>
#endif

__ESBMC_contract int ffs(int x)
{
  __ESBMC_requires(1);
  __ESBMC_ensures(__ESBMC_return_value >= 0);
  __ESBMC_ensures(__ESBMC_return_value <= (int)(8 * sizeof(int)));
  __ESBMC_ensures((x == 0) == (__ESBMC_return_value == 0));
  __ESBMC_ensures(
    (__ESBMC_return_value == 0) ||
    (((unsigned)x & (1u << (__ESBMC_return_value - 1))) != 0));
  __ESBMC_ensures(
    (__ESBMC_return_value <= 1) ||
    (((unsigned)x & ((1u << (__ESBMC_return_value - 1)) - 1u)) == 0));
  __ESBMC_assigns();
__ESBMC_HIDE:;
  if (x == 0)
    return 0;

  int pos = 1;
  // __contractor_loop: ffs:0
  __ESBMC_unroll(8 * sizeof(int) + 1);
  while ((x & 1) == 0)
  {
    x >>= 1;
    pos++;
  }

  return pos;
}
