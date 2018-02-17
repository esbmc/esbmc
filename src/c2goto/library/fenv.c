#include <fenv.h>

extern int __ESBMC_rounding_mode;

inline int fegetround(void)
{
__ESBMC_HIDE:;
  switch(__ESBMC_rounding_mode)
  {
#ifdef FE_DOWNWARD
  case 1:
    return FE_DOWNWARD;
#endif
  case 0:
    return FE_TONEAREST;
  case 3:
    return FE_TOWARDZERO;
#ifdef FE_UPWARD
  case 2:
    return FE_UPWARD;
#endif
  default:
    break;
  }
  return -1;
}

inline int fesetround(int rounding_mode)
{
__ESBMC_HIDE:;
  switch(rounding_mode)
  {
#ifdef FE_DOWNWARD
  case FE_DOWNWARD:
    __ESBMC_rounding_mode = 1;
    break;
#endif
  case FE_TONEAREST:
    __ESBMC_rounding_mode = 0;
    break;
  case FE_TOWARDZERO:
    __ESBMC_rounding_mode = 3;
    break;
#ifdef FE_UPWARD
  case FE_UPWARD:
    __ESBMC_rounding_mode = 2;
    break;
#endif
  default:
    break;
  }
  return 0; // we never fail
}
