
#include <setjmp.h>

// longjmp's transfer of control is lowered by goto-programs/lower_longjmp.cpp,
// which reads these after every call and matches the token against the one
// each setjmp call site recorded in its frame.
_Thread_local _Bool __ESBMC_longjmp_pending;
_Thread_local int __ESBMC_longjmp_value;
_Thread_local long __ESBMC_longjmp_token;
_Thread_local long __ESBMC_setjmp_count;

#undef setjmp
int setjmp(jmp_buf __env)
{
__ESBMC_HIDE:;
  __ESBMC_unreachable();
  *(long *)__env = ++__ESBMC_setjmp_count;
  return 0;
}

// Due to some macro expansion some programs may have the _setjmp instead
int _setjmp(jmp_buf __env)
{
__ESBMC_HIDE:;
  return setjmp(__env);
}

_Noreturn void longjmp(jmp_buf env, int status)
{
__ESBMC_HIDE:;
  __ESBMC_unreachable();
  __ESBMC_longjmp_token = *(long *)env;
  // C11 7.13.2.1p4: setjmp cannot return 0 through longjmp
  __ESBMC_longjmp_value = status ? status : 1;
  __ESBMC_longjmp_pending = 1;
}
