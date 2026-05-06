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

  unsigned x_orig = (unsigned)x;
  int pos = 1;
  // Invariant relates the current x to the original via the shift count
  // pos - 1, and pins the lower (pos - 1) bits of x_orig as zero. Together
  // with x_orig != 0 this forces termination at or before bit width.
  __ESBMC_loop_invariant(
      pos >= 1 && pos <= (int)(8 * sizeof(int))
      && (unsigned)x == (x_orig >> (pos - 1))
      && (x_orig & ((1u << (pos - 1)) - 1u)) == 0u);
  // __contractor_loop: ffs:0
  while ((x & 1) == 0)
  {
    x >>= 1;
    pos++;
  }

  return pos;
}
