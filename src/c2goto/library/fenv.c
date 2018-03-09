#include <fenv.h>

extern int __ESBMC_rounding_mode;

inline int fegetround(void)
{
__ESBMC_HIDE:;
  switch(__ESBMC_rounding_mode)
  {
  case 0:
    return FE_TONEAREST;
  case 2:
    return FE_UPWARD;
  case 3:
    return FE_DOWNWARD;
  case 4:
    return FE_TOWARDZERO;
  default:
    break;
  }
  return 0;
}

inline int fesetround(int rounding_mode)
{
__ESBMC_HIDE:;
  switch(rounding_mode)
  {
  case FE_TONEAREST:
    __ESBMC_rounding_mode = 0;
    break;
  case FE_UPWARD:
    __ESBMC_rounding_mode = 2;
    break;
  case FE_DOWNWARD:
    __ESBMC_rounding_mode = 3;
    break;
  case FE_TOWARDZERO:
    __ESBMC_rounding_mode = 4;
    break;
  default:
    break;
  }
  return 0; // we never fail
}
