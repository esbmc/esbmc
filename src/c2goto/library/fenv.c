#include <fenv.h>

int __ESBMC_rounding_mode = 0;

inline int fegetround(void)
{
__ESBMC_HIDE:;
  return __ESBMC_rounding_mode == 3   ? FE_DOWNWARD
         : __ESBMC_rounding_mode == 0 ? FE_TONEAREST
         : __ESBMC_rounding_mode == 4 ? FE_TOWARDZERO
         : __ESBMC_rounding_mode == 2 ? FE_UPWARD
                                      : -1;
}

/* A natively compiled program uses the host's FE_* macros, but a goto binary
 * produced on another architecture carries that platform's values: x86 encodes
 * the mode in bits 10-11 (FE_DOWNWARD 0x400, FE_UPWARD 0x800, FE_TOWARDZERO
 * 0xc00) and AArch64 in bits 22-23 with UPWARD and DOWNWARD swapped
 * (FE_UPWARD 0x400000, FE_DOWNWARD 0x800000, FE_TOWARDZERO 0xc00000). The two
 * encodings are disjoint, so recognising both is unambiguous. Without it an
 * unrecognised value fell through to the to-nearest default and --binary
 * verification of a foreign binary silently ignored every fesetround. */
#define __ESBMC_FE_X86_DOWNWARD 0x400
#define __ESBMC_FE_X86_UPWARD 0x800
#define __ESBMC_FE_X86_TOWARDZERO 0xc00
#define __ESBMC_FE_ARM_UPWARD 0x400000
#define __ESBMC_FE_ARM_DOWNWARD 0x800000
#define __ESBMC_FE_ARM_TOWARDZERO 0xc00000

inline int fesetround(int rounding_mode)
{
__ESBMC_HIDE:;
  __ESBMC_rounding_mode =
    (rounding_mode == FE_DOWNWARD || rounding_mode == __ESBMC_FE_X86_DOWNWARD ||
     rounding_mode == __ESBMC_FE_ARM_DOWNWARD)
      ? 3
    : rounding_mode == FE_TONEAREST ? 0
    : (rounding_mode == FE_TOWARDZERO ||
       rounding_mode == __ESBMC_FE_X86_TOWARDZERO ||
       rounding_mode == __ESBMC_FE_ARM_TOWARDZERO)
      ? 4
    : (rounding_mode == FE_UPWARD || rounding_mode == __ESBMC_FE_X86_UPWARD ||
       rounding_mode == __ESBMC_FE_ARM_UPWARD)
      ? 2
      : 0;
  return 0; // we never fail
}
