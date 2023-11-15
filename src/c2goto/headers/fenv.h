
/* their <fenv.h> contains inline function definitions incompatible with ours */
#ifdef __FreeBSD__

/* partially taken from FreeBSD-14 */

#pragma once

/* The high 32 bits contain fpcr, low 32 contain fpsr. */
typedef __UINT64_TYPE__ fenv_t;
typedef __UINT64_TYPE__ fexcept_t;

/* Exception flags */
#define FE_INVALID      0x00000001
#define FE_DIVBYZERO    0x00000002
#define FE_OVERFLOW     0x00000004
#define FE_UNDERFLOW    0x00000008
#define FE_INEXACT      0x00000010
#define FE_ALL_EXCEPT   (FE_DIVBYZERO | FE_INEXACT | \
                         FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW)

/*
 * Rounding modes
 *
 * We can't just use the hardware bit values here, because that would
 * make FE_UPWARD and FE_DOWNWARD negative, which is not allowed.
 */
#define FE_TONEAREST    0x0
#define FE_UPWARD       0x1
#define FE_DOWNWARD     0x2
#define FE_TOWARDZERO   0x3
#define _ROUND_MASK     (FE_TONEAREST | FE_DOWNWARD | \
                         FE_UPWARD | FE_TOWARDZERO)
#define _ROUND_SHIFT    22

/* Default floating-point environment */
extern const fenv_t     __ESBMC_fe_dfl_env;
#define FE_DFL_ENV      (&__ESBMC_fe_dfl_env)

int feclearexcept(int excepts);
int fegetexceptflag(fexcept_t *flagp, int excepts);
int feraiseexcept(int excepts);
int fesetexceptflag(const fexcept_t *flagp, int excepts);
int fetestexcept(int excepts);

int fegetround(void);
int fesetround(int rounding_mode);

int fegetenv(fenv_t *envp);
int feholdexcept(fenv_t *envp);
int fesetenv(const fenv_t *envp);
int feupdateenv(const fenv_t *envp);

#else /* !defined __FreeBSD__ */

/* For SV-COMP's pre-processed sources we need the system's definitions of
 * (at least) the constants for the rounding modes. */
#include_next <fenv.h>

#endif
