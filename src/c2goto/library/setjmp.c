
#include <setjmp.h>

#undef setjmp
int setjmp(jmp_buf __env)
{
__ESBMC_HIDE:;
  __ESBMC_unreachable();
  return nondet_int();
}

int _setjmp(jmp_buf __env)
{
__ESBMC_HIDE:;
  return setjmp(__env);
}

_Noreturn void longjmp(jmp_buf env, int status)
{
__ESBMC_HIDE:;
  __ESBMC_unreachable();
}
