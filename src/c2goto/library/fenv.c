#include <fenv.h>

extern int __ESBMC_rounding_mode;

inline int fegetround(void)
{
__ESBMC_HIDE:;
  return __ESBMC_rounding_mode == 3
           ? FE_DOWNWARD
           : __ESBMC_rounding_mode == 0
               ? FE_TONEAREST
               : __ESBMC_rounding_mode == 4
                   ? FE_TOWARDZERO
                   : __ESBMC_rounding_mode == 2 ? FE_UPWARD : -1;
}

inline int fesetround(int rounding_mode)
{
__ESBMC_HIDE:;
  __ESBMC_rounding_mode = rounding_mode == FE_DOWNWARD
                            ? 3
                            : rounding_mode == FE_TONEAREST
                                ? 0
                                : rounding_mode == FE_TOWARDZERO
                                    ? 4
                                    : rounding_mode == FE_UPWARD ? 2 : 0;
  return 0; // we never fail
}
