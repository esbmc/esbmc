/*
 * fenv.c
 *
 *  Created on: Aug 24, 2016
 *      Author: mramalho
 */

#include <fenv.h>

extern int __ESBMC_rounding_mode;

inline int fegetround(void)
{
  __ESBMC_HIDE:;
  return
         #ifdef FE_DOWNWARD
         __ESBMC_rounding_mode==1?FE_DOWNWARD:
         #endif
         __ESBMC_rounding_mode==0?FE_TONEAREST:
         __ESBMC_rounding_mode==3?FE_TOWARDZERO:
         #ifdef FE_UPWARD
         __ESBMC_rounding_mode==2?FE_UPWARD:
         #endif
         -1;
}

inline int fesetround(int rounding_mode)
{
  __ESBMC_HIDE:;
  __ESBMC_rounding_mode=
    #ifdef FE_DOWNWARD
    rounding_mode==FE_DOWNWARD?1:
    #endif
    rounding_mode==FE_TONEAREST?0:
    rounding_mode==FE_TOWARDZERO?3:
    #ifdef FE_UPWARD
    rounding_mode==FE_UPWARD?2:
    #endif
    0;
  return 0; // we never fail
}
